"""F1 tyre degradation prediction package."""

from .config import DEFAULT_MODEL_NAMES, PROJECT_ROOT
from .strict_feature_policy import validate_feature_names

__all__ = ["DEFAULT_MODEL_NAMES", "PROJECT_ROOT", "validate_feature_names"]
