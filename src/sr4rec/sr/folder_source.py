"""Method 3: a folder of pre-computed SR images (PNG, same relative paths as the dataset)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..errors import SRModelError
from ..utils import png_name, sha256_file, stable_key
from .base import SRSource

MAX_LISTED = 20


class FolderSR(SRSource):
    def __init__(self, name: str, folder: Path, scale: int):
        if not folder.is_dir():
            raise SRModelError(f"SR model {name!r}: images folder not found: {folder}")
        super().__init__(name=name, kind="images", scale=scale, description="images", display_file=f"{folder.name}/")
        self.folder = folder

    def file_for(self, path: str) -> Path:
        return self.folder / png_name(path)

    def check(self, expected: dict[str, tuple[int, int]], all_paths: set[str]) -> tuple[str, list[str]]:
        """Check the folder against ``expected`` (path -> LR (width, height)).

        Returns a status text for --dry-run and a list of warnings (extra files).
        Raises SRModelError for missing files and wrong sizes.
        """
        missing, wrong = [], []
        for path, (w, h) in expected.items():
            f = self.file_for(path)
            if not f.is_file():
                missing.append(png_name(path))
                continue
            try:
                with Image.open(f) as im:
                    size = im.size
            except OSError as err:
                wrong.append(f"{png_name(path)}: cannot be read ({err})")
                continue
            if size != (w * self.scale, h * self.scale):
                wrong.append(f"{png_name(path)}: {size[0]} x {size[1]} px, expected {w * self.scale} x {h * self.scale} px "
                             f"({self.scale} x the LR size {w} x {h})")
        known = {png_name(p) for p in all_paths}
        extra = sorted(str(f.relative_to(self.folder).as_posix()) for f in self.folder.rglob("*.png")
                       if f.relative_to(self.folder).as_posix() not in known)
        problems = []
        if missing:
            problems.append(f"{len(missing)} expected image(s) are missing, for example: " + ", ".join(missing[:MAX_LISTED]))
        if wrong:
            problems.append(f"{len(wrong)} image(s) have the wrong size:\n    " + "\n    ".join(wrong[:MAX_LISTED]))
        if problems:
            raise SRModelError(f"SR model {self.name!r} (images folder {self.folder}):\n  " + "\n  ".join(problems)
                               + "\nSee docs/adding_sr_models.md, method 3.")
        warnings = []
        if extra:
            warnings.append(f"SR model {self.name!r}: {len(extra)} PNG file(s) in {self.folder} do not match any dataset "
                            f"image and are ignored (for example {extra[0]}).")
        hashes = [(png_name(p), sha256_file(self.file_for(p))) for p in sorted(expected)]
        self.key = stable_key("images", hashes)
        self.files = {str(self.folder): self.key}
        return f"{len(expected):,}/{len(expected):,} images found", warnings
