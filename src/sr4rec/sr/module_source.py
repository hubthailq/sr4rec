"""Method 2: a user Python file with a torch.nn.Module class.

Interface: integer class attribute ``scale``; ``__init__(weights=None)``;
``forward(x)`` maps (N, 3, H, W) in [0, 1] to (N, 3, H*scale, W*scale).
Optional integer attribute ``size_multiple``: the input height and width are
padded to a multiple of it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import torch
from torch import nn

from ..errors import SRModelError
from ..utils import sha256_file
from .base import SizeRule, SRSource, run_padded, run_tiled


def _import_class(file: Path, class_name: str, name: str):
    if not file.is_file():
        raise SRModelError(f"SR model {name!r}: module file not found: {file}")
    mod_name = "sr4rec_user_" + hashlib.sha256(str(file).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        raise SRModelError(f"SR model {name!r}: {file} cannot be imported as a Python module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    sys.path.insert(0, str(file.parent))
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # noqa: BLE001
        raise SRModelError(f"SR model {name!r}: importing {file.name} failed ({type(err).__name__}: {err})") from None
    finally:
        if sys.path and sys.path[0] == str(file.parent):
            sys.path.pop(0)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise SRModelError(f"SR model {name!r}: class {class_name!r} is not defined in {file.name}.")
    return cls


class ModuleSR(SRSource):
    def __init__(self, name: str, module: str, base_resolve, scale: int, weights: Path | None, tile: int | None = None):
        file_part, class_name = module.rsplit(":", 1)
        file = base_resolve(file_part)
        cls = _import_class(file, class_name, name)
        if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
            raise SRModelError(f"SR model {name!r}: {class_name} is not a subclass of torch.nn.Module.")
        s = getattr(cls, "scale", None)
        if not isinstance(s, int) or isinstance(s, bool):
            raise SRModelError(f"SR model {name!r}: {class_name} must define an integer class attribute `scale`.")
        if s != scale:
            raise SRModelError(f"SR model {name!r}: {class_name}.scale is {s}, but the configuration uses scale {scale}.")
        if weights is not None and not weights.is_file():
            raise SRModelError(f"SR model {name!r}: weights file not found: {weights}")
        try:
            model = cls(weights=str(weights) if weights is not None else None)
        except Exception as err:  # noqa: BLE001
            raise SRModelError(f"SR model {name!r}: {class_name}(weights=...) failed ({type(err).__name__}: {err})") from None
        model.eval()
        files = {str(file): sha256_file(file)}
        if weights is not None:
            files[str(weights)] = sha256_file(weights)
        multiple = getattr(model, "size_multiple", 1)
        self.rule = SizeRule(multiple_of=int(multiple) if isinstance(multiple, int) and multiple > 0 else 1)
        self.model = model
        self.tile = tile
        super().__init__(
            name=name, kind="module", scale=scale,
            key="module:" + ":".join(f"{sha}" for sha in files.values()) + f":{class_name}:tile{tile}",
            files=files, description=f"module   {module}",
            display_file=(weights.name if weights is not None else file.name),
        )
        self._check_interface()

    @torch.no_grad()
    def _check_interface(self) -> None:
        x = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
        try:
            y = self.model(x)
        except Exception as err:  # noqa: BLE001
            raise SRModelError(f"SR model {self.name!r}: forward() failed on a (2, 3, 16, 16) test tensor "
                               f"({type(err).__name__}: {err})") from None
        expected = (2, 3, 16 * self.scale, 16 * self.scale)
        if not isinstance(y, torch.Tensor):
            raise SRModelError(f"SR model {self.name!r}: forward() must return a torch.Tensor (got {type(y).__name__}).")
        if tuple(y.shape) != expected:
            raise SRModelError(f"SR model {self.name!r}: forward() returned shape {tuple(y.shape)} for a (2, 3, 16, 16) "
                               f"input; expected {expected}.")
        if not y.is_floating_point():
            raise SRModelError(f"SR model {self.name!r}: forward() must return a floating-point tensor (got {y.dtype}).")
        if not torch.isfinite(y).all():
            raise SRModelError(f"SR model {self.name!r}: forward() returned NaN or infinite values.")

    def to(self, device: torch.device) -> ModuleSR:
        self.model.to(device)
        self.model.eval()
        return self

    @torch.no_grad()
    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        if self.tile:
            y = run_tiled(self.model, lr, self.scale, self.tile, self.rule)
        else:
            y = run_padded(self.model, lr, self.scale, self.rule)
        return y.float().clamp(0.0, 1.0)
