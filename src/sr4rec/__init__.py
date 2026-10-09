"""SR4Rec: measure whether super-resolution helps closed-set image recognition."""

import os as _os

# Operators that the Apple GPU backend does not implement run on the CPU instead of failing.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__", "Run", "load_config"]


def __getattr__(name):  # lazy imports keep `import sr4rec` light
    if name == "Run":
        from .pipeline import Run

        return Run
    if name == "load_config":
        from .config import load_config

        return load_config
    raise AttributeError(name)
