"""SR sources and the padding / tiling helpers."""

import os
import textwrap
from pathlib import Path

import pytest
import torch
from PIL import Image

from sr4rec.errors import SRModelError
from sr4rec.sr.base import SizeRule, pad_for, run_padded, run_tiled
from sr4rec.sr.bicubic import BicubicSR
from sr4rec.sr.folder_source import FolderSR
from sr4rec.sr.module_source import ModuleSR

GOOD = """
import torch
from torch import nn

class Net(nn.Module):
    scale = 4
    def __init__(self, weights=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode="nearest")
    def forward(self, x):
        return self.up(x)
"""


def module_file(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "net.py"
    f.write_text(textwrap.dedent(body))
    return f


def make(tmp_path, body, cls="Net", scale=4, weights=None):
    f = module_file(tmp_path, body)
    return ModuleSR("m", f"{f.name}:{cls}", lambda p: tmp_path / p, scale, weights)


def test_module_ok(tmp_path):
    src = make(tmp_path, GOOD)
    y = src.upscale(torch.rand(1, 3, 13, 17))
    assert y.shape == (1, 3, 52, 68) and src.kind == "module" and len(src.files) == 1


@pytest.mark.parametrize("body,cls,scale,message", [
    (GOOD.replace("    scale = 4\n", ""), "Net", 4, "integer class attribute `scale`"),
    (GOOD, "Net", 2, "scale is 4, but the configuration uses scale 2"),
    (GOOD.replace("(nn.Module)", "(object)"), "Net", 4, "not a subclass of torch.nn.Module"),
    (GOOD.replace("return self.up(x)", "return x"), "Net", 4, "returned shape"),
    (GOOD.replace("return self.up(x)", "return self.up(x) * float('nan')"), "Net", 4, "NaN"),
    (GOOD, "Missing", 4, "not defined"),
    (GOOD.replace("import torch", "import torch\nraise RuntimeError('boom')"), "Net", 4, "importing net.py failed"),
])
def test_module_interface_errors(tmp_path, body, cls, scale, message):
    with pytest.raises(SRModelError, match=message):
        make(tmp_path, body, cls, scale)


def test_module_file_missing(tmp_path):
    with pytest.raises(SRModelError, match="module file not found"):
        ModuleSR("m", "nope.py:Net", lambda p: tmp_path / p, 4, None)


def test_example_tiny_espcn_loads(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples" / "sr_models"
    src = ModuleSR("tiny", "tiny_espcn.py:TinyESPCN", lambda p: root / p, 4, None)
    assert src.upscale(torch.rand(1, 3, 16, 16)).shape == (1, 3, 64, 64)


def test_weights_scale_mismatch_stops(tmp_path):
    spandrel = pytest.importorskip("spandrel")  # noqa: F841
    from sr4rec.sr.pth_source import SpandrelSR

    w = Path(os.environ.get("SR4REC_TEST_WEIGHTS", "weights")) / "RealESRGAN_x4plus.pth"
    if not w.is_file():
        pytest.skip("set SR4REC_TEST_WEIGHTS to a folder with RealESRGAN_x4plus.pth to run this test")
    with pytest.raises(SRModelError, match="upscales x4, but the configuration uses scale 2"):
        SpandrelSR("r", w, 2)


def test_weights_unknown_architecture(tmp_path):
    from sr4rec.sr.pth_source import SpandrelSR

    f = tmp_path / "junk.pth"
    torch.save({"foo.weight": torch.zeros(3)}, f)
    with pytest.raises(SRModelError, match="could not detect the architecture|could not be loaded"):
        SpandrelSR("j", f, 4)


def test_folder_source_checks(tmp_path):
    folder = tmp_path / "sr"
    (folder / "a").mkdir(parents=True)
    Image.new("RGB", (40, 32)).save(folder / "a" / "x.png")
    Image.new("RGB", (41, 32)).save(folder / "a" / "y.png")
    Image.new("RGB", (8, 8)).save(folder / "extra.png")
    src = FolderSR("f", folder, 4)
    expected = {"a/x.jpg": (10, 8), "a/y.jpg": (10, 8), "a/z.jpg": (10, 8)}
    with pytest.raises(SRModelError) as err:
        src.check(expected, set(expected))
    text = str(err.value)
    assert "1 expected image(s) are missing" in text and "a/z.png" in text
    assert "a/y.png: 41 x 32 px, expected 40 x 32 px" in text
    status, warnings = FolderSR("f", folder, 4).check({"a/x.jpg": (10, 8)}, {"a/x.jpg", "a/y.jpg"})
    assert status.startswith("1/1") and "extra.png" in warnings[0]


def test_folder_missing():
    with pytest.raises(SRModelError, match="images folder not found"):
        FolderSR("f", Path("/nonexistent/sr"), 4)


def test_padding_meets_rule_and_crops_back():
    x = torch.rand(1, 3, 5, 3)
    padded, h, w = pad_for(x, SizeRule(minimum=16, multiple_of=8))
    assert padded.shape[-2:] == (16, 16) and (h, w) == (5, 3)
    y = run_padded(lambda t: torch.nn.functional.interpolate(t, scale_factor=2), x, 2, SizeRule(minimum=16))
    assert y.shape == (1, 3, 10, 6) and torch.allclose(y[..., ::2, ::2], x)


def test_tiling_matches_whole_image_for_local_model():
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")  # noqa: E731
    x = torch.rand(1, 3, 150, 90)
    whole = run_padded(fn, x, 4, SizeRule())
    tiled = run_tiled(fn, x, 4, 64, SizeRule())
    assert torch.allclose(whole, tiled, atol=1e-6)


def test_bicubic_shape():
    assert BicubicSR(3).upscale(torch.rand(1, 3, 7, 5)).shape == (1, 3, 21, 15)
