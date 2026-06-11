from .base import ToolCallResponse, ToolCallResult, VLMBackend
from .slowburn_api import SlowBurnAPIBackend  # noqa: F401 — triggers Registry registration

try:
    from .vllm import VLLMQwenBackend  # noqa: F401 — triggers Registry registration
except ImportError:
    pass

__all__ = [
    "ToolCallResponse",
    "ToolCallResult",
    "VLMBackend",
]
