"""Base types and abstract interface for all VLM backends.

``VLMBackend`` is a ``Typed + Registry`` base class.  Concrete backends
register themselves via ``aliases`` and are instantiated with
``VLMBackend.of("api", litellm_model=..., ...)`` — no factory function needed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from morphic import Registry, Typed

# ---------------------------------------------------------------------------
#  Tool-calling response types (used by the agentic diagnostic loop)
# ---------------------------------------------------------------------------


class ToolCallResult(Typed):
    """One tool call extracted from a VLM response.

    Represents the structured output of a native function-calling
    response (``response.choices[0].message.tool_calls[i]``).  The
    ``arguments`` dict is already parsed from the JSON string.
    """

    id: str
    name: str
    arguments: Dict[str, Any]

    def __str__(self) -> str:
        args_str: str = (
            ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
            if len(self.arguments) > 0
            else "(no args)"
        )
        return f"{self.name}({args_str}) [id={self.id}]"


class ToolCallResponse(Typed):
    """Structured response from a VLM call that may contain tool calls.

    Used by ``VLMBackend.call_for_tool()`` (Phase 1 of the agentic tool
    loop).  The VLM either returns tool_calls (diagnostic request) or
    content only (VLM declined to call a tool).
    """

    content: Optional[str] = None
    tool_calls: List[ToolCallResult] = []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        lines: List[str] = []
        if self.content is not None:
            lines.append(f"  content: {self.content}")
        else:
            lines.append("  content: (none)")
        if len(self.tool_calls) > 0:
            lines.append(f"  tool_calls ({len(self.tool_calls)}):")
            for tc in self.tool_calls:
                lines.append(f"    → {tc}")
        else:
            lines.append("  tool_calls: (none)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  VLMBackend base class
# ---------------------------------------------------------------------------


class VLMBackend(Typed, Registry, ABC):
    """Unified interface that every VLM provider must implement.

    Consumers instantiate via the Registry::

        backend = VLMBackend.of("api", litellm_model="azure/gpt-5-mini", ...)

    and call ``backend.call(prompt=..., ...)`` for all LLM interactions.

    Three call methods:
        - ``call()`` — single text/Typed response (proposals, code-gen, summaries)
        - ``call_batch()`` — multiple independent calls in parallel
        - ``call_for_tool()`` — raw tool_calls response (Phase 1: diagnostic loop)

    Verbosity levels:
        0 -- silent (errors only)
        1 -- show parsed results: description, families, model code
        2 -- also show raw prompt, raw response, full message list
    """

    @abstractmethod
    def call(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_type: Optional[Type[Typed]] = None,
        verbosity: int,
        dataset_prefix: str = "",
    ) -> Union[str, Typed]:
        """Run one LLM call with optional images, tools, and response parsing.

        Args:
            prompt: Text prompt (prediction, feedback, code-gen, summary, etc.).
            images: Image paths or URLs to include.  ``None`` for text-only.
            tools: OpenAI-format tool schemas for function calling.
                ``None`` disables tool calling for this request.
            response_type: A ``Typed`` subclass to parse the response into.
                When provided, the raw VLM text is parsed as JSON and
                constructed into the given Typed class via Pydantic coercion.
                On parse/validation failure, the backend retries via
                SlowBurn's built-in retry mechanism.  When ``None``, the
                raw response string is returned without parsing.
            verbosity: Logging level (0=silent, 1=parsed, 2=debug).
            dataset_prefix: Optional dataset marker for log identification.

        Returns:
            When ``response_type`` is ``None``: the raw response ``str``.
            When ``response_type`` is provided: a validated instance of
            that Typed class.

        Raises:
            ValueError: If response parsing/validation fails after all
                retries.  The prompt, images, and error are logged before
                raising.  Callers must handle this — the backend never
                silently returns ``None``.
        """
        ...

    @abstractmethod
    def call_batch(
        self,
        *,
        prompts: List[str],
        images_per_prompt: Optional[List[Optional[List[Union[str, Path]]]]] = None,
        response_type: Optional[Type[Typed]] = None,
        verbosity: int,
        dataset_prefix: str = "",
    ) -> List[Union[str, Typed]]:
        """Run multiple independent LLM calls concurrently.

        Each prompt is an independent call (no shared conversation state).
        Backends that support concurrent execution (e.g. SlowBurn) will
        fire all calls in parallel.  Backends without concurrency support
        (e.g. vLLM local) fall back to sequential execution.

        Args:
            prompts: List of text prompts to send.
            images_per_prompt: Optional list, same length as ``prompts``,
                where each element is ``None`` (text-only) or a list of
                image paths.  If ``None``, all calls are text-only.
            response_type: A ``Typed`` subclass to parse each response into.
                Applied identically to all calls.  ``None`` returns raw strings.
            verbosity: Logging level (0=silent, 1=parsed, 2=debug).
            dataset_prefix: Optional dataset marker for log identification.

        Returns:
            List of results in the same order as ``prompts``.  Each element
            is either a ``str`` or a ``Typed`` instance, depending on
            ``response_type``.  Failed calls raise — the batch does NOT
            return partial results.
        """
        ...

    @abstractmethod
    def call_for_tool(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        tools: List[Dict[str, Any]],
        tool_choice: str,
        verbosity: int,
        dataset_prefix: str = "",
    ) -> ToolCallResponse:
        """Send a tool-calling request and return raw tool_calls + content.

        Unlike ``call()``, this method does NOT use a validator or
        ``response_type``.  It returns the raw VLM response decomposed
        into ``content`` and ``tool_calls``.  Used for Phase 1
        (diagnostic) of the agentic tool loop.

        Args:
            prompt: Context prompt describing the current fit state.
            images: Image paths (fit overlay, histogram, etc.).
            tools: OpenAI-format tool schemas.  Must be non-empty.
            tool_choice: ``"auto"`` (VLM decides) or ``"required"``
                (VLM must call at least one tool).
            verbosity: Logging level.
            dataset_prefix: Optional dataset marker for log identification.

        Returns:
            ``ToolCallResponse`` with ``content`` (if the VLM produced
            text) and ``tool_calls`` (if the VLM called a function).
            Both can be populated (Claude), or only one (GPT/Qwen).
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Shut down the backend and release resources."""
        ...

    @abstractmethod
    def get_cost_report(self) -> str:
        """Return a human-readable cost report string."""
        ...
