"""Recognition backbones from timm."""

from __future__ import annotations

import warnings

import timm
from torch import nn

from ..config import VALIDATED_BACKBONES
from ..errors import MissingResource, SR4RecError


def check_backbone_name(name: str) -> str | None:
    """Return a warning for names outside the validated list; raise for unknown names."""
    if not timm.is_model(name.split(".")[0]):
        raise SR4RecError(f"Backbone {name!r} is not a timm model name. Validated backbones: {', '.join(VALIDATED_BACKBONES)}.")
    if name not in VALIDATED_BACKBONES:
        return f"Backbone {name!r} is experimental: it is not in the validated list ({', '.join(VALIDATED_BACKBONES)})."
    return None


def create_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    except Exception as err:  # noqa: BLE001 - network errors come in many types
        if pretrained:
            raise MissingResource(
                f"The ImageNet-pretrained weights of {name!r} could not be obtained ({type(err).__name__}: {err}). "
                "timm downloads them from Hugging Face on first use; connect to the internet once or fill the "
                "Hugging Face cache (HF_HOME) in advance, see the README section 'Offline use'. "
                "Setting recognizer.pretrained: false avoids the download (lower accuracy)."
            ) from None
        raise SR4RecError(f"Backbone {name!r} could not be created: {err}") from None


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_state(model: nn.Module, state: dict) -> nn.Module:
    model.load_state_dict(state)
    return model.eval()
