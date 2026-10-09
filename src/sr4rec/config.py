"""Configuration schema of SR4Rec.

This module is the single source of truth for every configuration key: its
type, default value, accepted values and documentation. The commented
``sr4rec.yaml`` written by ``sr4rec init`` and ``docs/config_reference.md`` are
both generated from these models (see :mod:`sr4rec.config_docs`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError

NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
RESERVED_SR_NAMES = {"bicubic", "hr"}
VALIDATED_BACKBONES = ("resnet18", "mobilenetv3_small_100", "convnext_tiny")
CHECK_MARKER = "[CHECK]"


def _doc(doc: str, type_: str, allowed: str | list[str], required: str = "no", **extra: Any) -> dict:
    """Build the documentation block attached to a schema field."""
    return {"doc": doc, "type": type_, "allowed": allowed, "required": required, **extra}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class SplitRatios(_Strict):
    train: float = Field(0.7, json_schema_extra=_doc("Fraction of images in the train split.", "float", "greater than 0 and less than 1"))
    val: float = Field(0.1, json_schema_extra=_doc("Fraction of images in the val split.", "float", "greater than 0 and less than 1"))
    test: float = Field(0.2, json_schema_extra=_doc("Fraction of images in the test split.", "float", "greater than 0 and less than 1"))

    @field_validator("train", "val", "test")
    @classmethod
    def _open_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("must be greater than 0 and less than 1")
        return v

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitRatios:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train + val + test must sum to 1.0 (got {total:.6f})")
        return self


class SplitConfig(_Strict):
    source: Literal["auto", "column", "ratios"] = Field(
        "auto",
        json_schema_extra=_doc(
            "Where the train/val/test assignment comes from.",
            "enum",
            [
                "auto   : use the `split` column of labels.csv if it exists, otherwise split by `ratios`",
                "column : use the `split` column; stop with an error if it is missing",
                "ratios : ignore any `split` column and split by `ratios`",
            ],
        ),
    )
    ratios: SplitRatios = Field(
        default_factory=SplitRatios,
        json_schema_extra=_doc(
            "Fractions of images for each split, used when SR4Rec creates the split. Applied inside every class "
            "(each class keeps at least one image in train, val and test). The report prints the actual counts.",
            "mapping with keys train, val, test",
            "each value greater than 0 and less than 1; the three values must sum to 1.0",
            default_repr="{train: 0.7, val: 0.1, test: 0.2}",
        ),
    )
    val_from_train: float = Field(
        0.1,
        json_schema_extra=_doc(
            "Fraction of the train images moved to val when the `split` column has train and test rows but no val rows.",
            "float",
            "greater than 0, at most 0.5",
        ),
    )
    seed: int = Field(
        0,
        json_schema_extra=_doc(
            "Seed used to create the split. It is independent of the recognizer seeds, so all recognizer seeds share the same split.",
            "integer",
            "0 to 2147483647",
        ),
    )

    @field_validator("val_from_train")
    @classmethod
    def _vft(cls, v: float) -> float:
        if not 0.0 < v <= 0.5:
            raise ValueError("must be greater than 0 and at most 0.5")
        return v

    @field_validator("seed")
    @classmethod
    def _seed(cls, v: int) -> int:
        if not 0 <= v <= 2**31 - 1:
            raise ValueError("must be between 0 and 2147483647")
        return v


class SRModelConfig(_Strict):
    name: str = Field(..., json_schema_extra=_doc(
        "Name used in reports.", "string", 'letters, digits, "_" and "-"; must be unique; "bicubic" and "hr" are reserved', required="yes"))
    weights: str | None = Field(None, json_schema_extra=_doc(
        "Method 1: a .pth or .safetensors file whose architecture spandrel can detect (its scale must equal `scale`). "
        "Method 2: optional file passed to the class as __init__(weights=<path>). Recommended location: weights/.",
        "path", "any existing file", required="conditional (method 1; optional in method 2)"))
    module: str | None = Field(None, json_schema_extra=_doc(
        "Method 2: a Python file and a torch.nn.Module class with an integer attribute `scale` equal to `scale` and "
        "forward(x) mapping (N, 3, H, W) in [0, 1] to (N, 3, H*scale, W*scale). Recommended location: sr_models/.",
        'string "<file.py>:<ClassName>"', "an existing Python file and a class defined in it", required="conditional (method 2)"))
    images: str | None = Field(None, json_schema_extra=_doc(
        "Method 3: a folder of PNG images with the same relative paths as the dataset images (train, val and test "
        "images for protocol `matched`; test images only for `fixed_recognizer`). Recommended location: sr_images/<name>/. "
        "CPU latency is not measured for this source.",
        "path", "an existing folder", required="conditional (method 3; no other source key)"))
    tile: int | None = Field(None, json_schema_extra=_doc(
        "LR tile size in pixels for tiled inference (tiles overlap by 16 px). Use it only if the GPU runs out of memory. "
        "Valid for methods 1 and 2 only.",
        "integer or null", "null (process the whole image) | 64 to 1024"))

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.match(NAME_PATTERN, v):
            raise ValueError('only letters, digits, "_" and "-" are allowed')
        if v.lower() in RESERVED_SR_NAMES:
            raise ValueError(f'"{v}" is reserved')
        return v

    @field_validator("module")
    @classmethod
    def _module(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^.+\.py:[A-Za-z_][A-Za-z0-9_]*$", v):
            raise ValueError('must look like "path/to/file.py:ClassName"')
        return v

    @field_validator("tile")
    @classmethod
    def _tile(cls, v: int | None) -> int | None:
        if v is not None and not 64 <= v <= 1024:
            raise ValueError("must be null or between 64 and 1024")
        return v

    @model_validator(mode="after")
    def _one_source(self) -> SRModelConfig:
        if self.images is not None:
            if self.weights is not None or self.module is not None:
                raise ValueError("`images` cannot be combined with `weights` or `module`")
            if self.tile is not None:
                raise ValueError("`tile` is only valid with `weights` or `module`")
        elif self.module is None and self.weights is None:
            raise ValueError("one source is required: `weights` (method 1), `module` (method 2) or `images` (method 3)")
        return self

    @property
    def method(self) -> str:
        if self.images is not None:
            return "images"
        if self.module is not None:
            return "module"
        return "weights"


class RecognizerConfig(_Strict):
    backbones: list[str] = Field(
        default_factory=lambda: list(VALIDATED_BACKBONES),
        json_schema_extra=_doc(
            "Recognition backbones (timm model names, fine-tuned for the dataset classes).",
            "list of strings",
            [
                "validated by SR4Rec: resnet18 | mobilenetv3_small_100 | convnext_tiny",
                'any other timm model name runs with an "experimental" warning',
            ],
            default_repr="[resnet18, mobilenetv3_small_100, convnext_tiny]",
        ),
    )
    pretrained: bool = Field(True, json_schema_extra=_doc(
        "Start from ImageNet-pretrained weights downloaded by timm (needs internet access or a filled HF_HOME cache).",
        "boolean", "true | false"))
    input_size: int = Field(224, json_schema_extra=_doc(
        "Side length of the square recognizer input. Every image is letterboxed (longer side resized to this "
        "value, aspect ratio kept, ImageNet-mean padding; never stretched) and ImageNet-normalized.",
        "integer", "64 to 512, multiple of 32"))
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], json_schema_extra=_doc(
        "Random seeds. One recognizer is trained per image pipeline, backbone and seed.",
        "list of integers", "1 to 10 distinct integers from 0 to 2147483647 (3 or more recommended)", default_repr="[0, 1, 2]"))

    @field_validator("backbones")
    @classmethod
    def _backbones(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one backbone is required")
        if len(set(v)) != len(v):
            raise ValueError("backbone names must be distinct")
        return v

    @field_validator("input_size")
    @classmethod
    def _input_size(cls, v: int) -> int:
        if not (64 <= v <= 512 and v % 32 == 0):
            raise ValueError("must be between 64 and 512 and a multiple of 32")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[int]) -> list[int]:
        if not 1 <= len(v) <= 10:
            raise ValueError("between 1 and 10 seeds are required")
        if len(set(v)) != len(v):
            raise ValueError("seeds must be distinct")
        if any(not 0 <= s <= 2**31 - 1 for s in v):
            raise ValueError("every seed must be between 0 and 2147483647")
        return v


class TrainingConfig(_Strict):
    epochs: int = Field(30, json_schema_extra=_doc(
        "Number of training epochs. The checkpoint with the best val Rank-1 accuracy is kept.", "integer", "1 to 500"))
    batch_size: int = Field(64, json_schema_extra=_doc("Mini-batch size.", "integer", "1 to 1024"))
    learning_rate: float = Field(3.0e-4, json_schema_extra=_doc(
        "AdamW learning rate (cosine schedule, 1 warm-up epoch).", "float", "greater than 0, at most 1.0"))
    weight_decay: float = Field(0.05, json_schema_extra=_doc("AdamW weight decay.", "float", "0.0 to 1.0"))
    horizontal_flip: bool = Field(False, json_schema_extra=_doc(
        "Random horizontal flip during training. Off by default because flipping can remove identity cues "
        "(e.g. left vs right ear).", "boolean", "true | false"))
    num_workers: int = Field(4, json_schema_extra=_doc("Data-loading worker processes.", "integer", "0 to 64"))

    @field_validator("epochs")
    @classmethod
    def _epochs(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError("must be between 1 and 500")
        return v

    @field_validator("batch_size")
    @classmethod
    def _bs(cls, v: int) -> int:
        if not 1 <= v <= 1024:
            raise ValueError("must be between 1 and 1024")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _lr(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("must be greater than 0 and at most 1.0")
        return v

    @field_validator("weight_decay")
    @classmethod
    def _wd(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v

    @field_validator("num_workers")
    @classmethod
    def _nw(cls, v: int) -> int:
        if not 0 <= v <= 64:
            raise ValueError("must be between 0 and 64")
        return v


class LatencyConfig(_Strict):
    enabled: bool = Field(True, json_schema_extra=_doc(
        "Measure CPU latency of each SR model, each backbone and the full pipeline.", "boolean", "true | false"))
    lr_size: int = Field(64, json_schema_extra=_doc("Side length of the square LR input used for timing.", "integer", "16 to 512"))
    threads: int = Field(4, json_schema_extra=_doc("Number of CPU threads used for timing.", "integer", "1 to 64"))

    @field_validator("lr_size")
    @classmethod
    def _lr_size(cls, v: int) -> int:
        if not 16 <= v <= 512:
            raise ValueError("must be between 16 and 512")
        return v

    @field_validator("threads")
    @classmethod
    def _threads(cls, v: int) -> int:
        if not 1 <= v <= 64:
            raise ValueError("must be between 1 and 64")
        return v


class ComparisonsConfig(_Strict):
    enabled: bool = Field(True, json_schema_extra=_doc(
        "Export one PNG per selected test image that combines the LR input, bicubic, every SR model and (synthetic "
        "mode) the HR image, each with an English caption: method name, predicted label with a correct/wrong mark, "
        "and PSNR/SSIM (synthetic mode only).", "boolean", "true | false"))
    count: int = Field(24, json_schema_extra=_doc("Number of test images to export.", "integer", "1 to 1000"))
    selection: Literal["mixed", "random", "all"] = Field("mixed", json_schema_extra=_doc(
        "How the test images are chosen.", "enum", [
            "mixed  : one third corrected by at least one SR model (bicubic wrong, SR right), one third degraded by at "
            "least one SR model (bicubic right, SR wrong), one third random; fixed selection seed 0",
            "random : uniformly at random from the test set; fixed selection seed 0",
            "all    : every test image (ignores `count`; can produce many files)",
        ]))
    backbone: str | None = Field(None, json_schema_extra=_doc(
        "Backbone whose predictions (first seed) are written in the captions.", "string or null",
        "null (first backbone in `recognizer.backbones`) | any name listed in `recognizer.backbones`"))

    @field_validator("count")
    @classmethod
    def _count(cls, v: int) -> int:
        if not 1 <= v <= 1000:
            raise ValueError("must be between 1 and 1000")
        return v


class RuntimeConfig(_Strict):
    device: Literal["auto", "cuda", "mps", "cpu"] = Field("auto", json_schema_extra=_doc(
        "Device for SR inference and recognizer training. If the GPU runs out of memory, the run never stops: "
        "SR switches to smaller tiles, training to gradient accumulation (same effective batch size) and "
        "evaluation to smaller batches, and as a last resort the step continues on the CPU; the report "
        "lists every such fallback.", "enum", [
            "auto : CUDA GPU if available, otherwise Apple GPU (MPS), otherwise the CPU",
            "cuda : CUDA GPU; if none is found, fall back to MPS, then the CPU, with a warning",
            "mps  : Apple GPU (Apple silicon); if unavailable, fall back to the CPU with a warning",
            "cpu  : always the CPU (slow; intended for small datasets and tests)",
        ]))
    output_dir: str = Field("runs", json_schema_extra=_doc("Parent folder for all runs.", "path", "any folder path"))
    cache_dir: str | None = Field(None, json_schema_extra=_doc(
        "Folder for cached LR/SR images and trained recognizer checkpoints, shared by all runs so that repeated runs "
        "are fast. Deleting it is always safe.", "path or null", "null (use <output_dir>/.cache) | any folder path"))
    run_name: str | None = Field(None, json_schema_extra=_doc(
        "Name of this run's folder inside `output_dir`.", "string or null",
        'null (automatic: <dataset>_<YYYY-MM-DD>_<HHMM>) | letters, digits, "_" and "-"'))

    @field_validator("run_name")
    @classmethod
    def _run_name(cls, v: str | None) -> str | None:
        if v is not None and not re.match(NAME_PATTERN, v):
            raise ValueError('only letters, digits, "_" and "-" are allowed')
        return v


class Config(_Strict):
    dataset: str = Field(..., json_schema_extra=_doc(
        "Dataset folder. It must contain `images/` and `labels.csv` (see docs/preparing_dataset.md). A relative path "
        "is resolved from the folder that contains this file. Recommended location: data/<dataset_name>.",
        "path", "an existing folder", required="yes", section="1. Dataset and protocol"))
    mode: Literal["synthetic", "native-lr"] = Field(..., json_schema_extra=_doc(
        "Resolution of the images in `images/`.", "enum", [
            "synthetic : images are high resolution (HR). SR4Rec downsamples them by `scale`; PSNR/SSIM and an HR "
            "upper bound are reported.",
            "native-lr : images are already low resolution. No downsampling; PSNR/SSIM are not available "
            "(there is no HR reference).",
        ], required="yes"))
    scale: Literal[2, 3, 4] = Field(4, json_schema_extra=_doc(
        "Upscaling factor. In `synthetic` mode it is also the downsampling factor. Every SR model below must have "
        "exactly this scale, otherwise the run stops.", "integer (enum)", "2 | 3 | 4"))
    protocol: Literal["matched", "fixed_recognizer"] = Field("matched", json_schema_extra=_doc(
        "How the recognizer is trained with respect to SR.", "enum", [
            "matched          : for every SR model (and bicubic), the train, val and test images all pass through "
            "that SR model; one recognizer is trained per SR model, backbone and seed. Answers: \"does a system "
            "built with this SR model recognize better?\"",
            "fixed_recognizer : one recognizer per backbone and seed, trained on HR images (synthetic) or "
            "bicubic-upscaled images (native-lr); only the test images pass through SR. Answers: \"does adding SR "
            "in front of an existing recognizer help?\" Fewer recognizer trainings.",
        ]))
    split: SplitConfig = Field(default_factory=SplitConfig, json_schema_extra={"section": "2. Train / val / test split"})
    sr: list[SRModelConfig] = Field(default_factory=list, json_schema_extra={"section": "3. SR models"})
    recognizer: RecognizerConfig = Field(default_factory=RecognizerConfig, json_schema_extra={"section": "4. Recognizer"})
    training: TrainingConfig = Field(default_factory=TrainingConfig, json_schema_extra={"section": "5. Training (recognizer only; SR models are never trained)"})
    latency: LatencyConfig = Field(default_factory=LatencyConfig, json_schema_extra={"section": "6. CPU latency benchmark"})
    comparisons: ComparisonsConfig = Field(default_factory=ComparisonsConfig, json_schema_extra={"section": "7. Visual comparisons (before SR / after SR)"})
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, json_schema_extra={"section": "8. Runtime"})

    @model_validator(mode="after")
    def _cross_checks(self) -> Config:
        names = [m.name for m in self.sr]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"SR model names must be unique; duplicated: {sorted(dup)}")
        cb = self.comparisons.backbone
        if cb is not None and cb not in self.recognizer.backbones:
            raise ValueError(f"comparisons.backbone {cb!r} is not listed in recognizer.backbones")
        return self


class LoadedConfig:
    """A validated configuration together with the folder its relative paths refer to."""

    def __init__(self, config: Config, base_dir: Path, source_file: Path | None, raw_text: str | None):
        self.config = config
        self.base_dir = base_dir
        self.source_file = source_file
        self.raw_text = raw_text

    def resolve(self, path: str | None) -> Path | None:
        if path is None:
            return None
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def unconfirmed_lines(text: str) -> list[tuple[int, str]]:
    """Return (line number, line) of every value line that still carries a [CHECK] marker.

    Comment-only lines (which explain what the marker means) are ignored.
    """
    found = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if CHECK_MARKER in line:
            found.append((i, line.rstrip()))
    return found


def _format_validation_error(err: ValidationError) -> str:
    lines = ["The configuration is invalid:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "(top level)"
        msg = e["msg"]
        if e["type"] == "extra_forbidden":
            msg = "unknown key (check the spelling; see docs/config_reference.md)"
        elif e["type"] == "literal_error":
            expected = e.get("ctx", {}).get("expected", "")
            values = [a or b for a, b in re.findall(r"'([^']*)'|\b(\d+)\b", expected)] or [expected]
            msg = f"{e.get('input')!r} is not allowed. Allowed values: {', '.join(values)}"
        elif msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def parse_config(data: dict, base_dir: Path, source_file: Path | None = None, raw_text: str | None = None) -> LoadedConfig:
    if not isinstance(data, dict):
        raise ConfigError("The configuration must be a YAML mapping (key: value pairs).")
    try:
        cfg = Config.model_validate(data)
    except ValidationError as err:
        raise ConfigError(_format_validation_error(err)) from None
    return LoadedConfig(cfg, base_dir, source_file, raw_text)


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path} is not valid YAML: {err}") from None
    return parse_config(data or {}, path.parent, path, text)


def config_to_dict(cfg: Config) -> dict:
    return cfg.model_dump(mode="json")
