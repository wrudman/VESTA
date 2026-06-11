"""SlowBurn/LiteLLM backend — a PURE TRANSPORT LAYER for LLM calls.

===========================================================================
SCOPE (READ THIS BEFORE MODIFYING — LLM AGENTS ESPECIALLY)
===========================================================================

This file is the adapter between VESTA's ``VLMBackend``
interface and the SlowBurn library (which wraps LiteLLM).  Its scope is
EXACTLY three things:

    1. **Configure** a SlowBurn LLM worker (model, tokens, temperature,
       retry count, cost budget) from the caller's config fields.
    2. **Translate** ``backend.call(prompt, images, tools, response_type,
       verbosity)`` into ``self._llm.call_llm_batch(prompts, ...)``.
    3. **Construct a validator** (when ``response_type`` is provided) that
       converts the raw response string into the caller's Typed class via
       ``parse_json_from_text`` + ``response_type(**parsed)``.

THIS FILE HAS ZERO KNOWLEDGE OF:
    - Distribution families, kernels, proposals, priors, descriptions,
      or any other domain concept.
    - What the VLM is being asked to do (fitting, code-gen, summarization).
    - What fields the response should contain — that is the domain of the
      ``response_type`` Typed class, whose Pydantic validators enforce the
      schema automatically.

DO NOT ADD:
    - Checks for specific JSON keys (``"proposals"``, ``"description"``,
      ``"tool_calls"``, ``"code"``, etc.).
    - Domain-specific error messages ("no model proposals", etc.).
    - Any logic that inspects the parsed dict before passing it to
      ``response_type(**parsed)``.

If the VLM returns a malformed response (wrong keys, missing fields,
tool-only with no content), ``response_type(**parsed)`` will raise
``ValueError`` (via Morphic/Pydantic validation), and SlowBurn will
retry automatically.  The backend never needs to know WHY it failed.

Provider routing is done entirely through the LiteLLM model string:
    - ``azure/gpt-5-mini``             -> Azure OpenAI
    - ``azure/responses/gpt-5-mini``   -> Azure OpenAI (Responses API)
    - ``bedrock/anthropic.claude-sonnet-4-20250514-v1:0`` -> AWS Bedrock
    - ``together_ai/Qwen/Qwen3.5-9B`` -> Together AI
    - ``openrouter/qwen/qwen3-vl-8b-instruct`` -> OpenRouter
    - ``gpt-4o-mini``                  -> OpenAI direct

API keys are read from standard env vars that LiteLLM recognizes:
    AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
    TOGETHERAI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

import litellm
from morphic import Typed, validate
from morphic.string import format_exception_msg
from pydantic import PrivateAttr
from slowburn import create_llm

from vesta.core.logging_utils import format_log_block

from .base import ToolCallResponse, ToolCallResult, VLMBackend
from .parsing import parse_json_from_text

logger: logging.Logger = logging.getLogger("vlm_backends.slowburn_api")


# ---------------------------------------------------------------------------
#  Typed-model validator factory
# ---------------------------------------------------------------------------


def _build_typed_validator(
    *,
    response_type: Type[Typed],
    verbosity: int,
) -> Callable[[str], Typed]:
    """Build a SlowBurn ``validator``: raw text -> Typed instance.

    This function is DOMAIN-AGNOSTIC.  It does not inspect the parsed
    dict's keys or values.  It delegates ALL schema validation to the
    ``response_type`` class's Pydantic field definitions and
    ``@field_validator`` methods.

    Steps:
        1. ``parse_json_from_text(response_text)`` — extract JSON.
        2. ``response_type(**parsed)`` — Pydantic validates and coerces.
        3. Return the Typed instance.

    If step 1 or 2 raises, the exception propagates as ``ValueError``
    and SlowBurn retries the LLM call automatically.

    DO NOT add key-specific checks here (e.g. checking for "proposals",
    "tool_calls", "description").  The ``response_type`` Typed class
    handles that via its own field definitions and validators.
    """

    def _validate(response_text: str) -> Typed:
        parsed: Dict[str, Any] = parse_json_from_text(response_text)

        try:
            result: Typed = response_type(**parsed)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"VLM response does not match {response_type.__name__} schema: "
                f"{format_exception_msg(exc)}. "
                f"Parsed keys: {list(parsed.keys())}"
            ) from exc

        if verbosity >= 2:
            logger.debug(
                f"Validated {response_type.__name__} (full parsed response logged by experiments layer)"
            )

        return result

    return _validate


def _resolve_images(images: Optional[List[Union[str, Path]]]) -> List[Path]:
    """Validate and resolve image paths, raising on missing/empty files."""
    if images is None:
        return []
    resolved: List[Path] = []
    for img in images:
        img_path: Path = Path(img)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if img_path.stat().st_size == 0:
            raise ValueError(f"Image file is empty: {img_path}")
        resolved.append(img_path)
    return resolved


def _format_reasoning_trace(
    *,
    model_id: str,
    litellm_params: Optional[Dict[str, Any]],
    tool_choice: Optional[str],
    assistant_message: Optional[Dict[str, Any]],
) -> str:
    """Format all reasoning-relevant fields from the assistant message.

    SlowBurn's ``call_llm`` returns ``model_dump(exclude_none=True)`` of
    LiteLLM's ``Message`` Pydantic model, so the assistant dict includes
    every provider-specific reasoning field that LiteLLM extracted.
    """
    lines: List[str] = [
        f"Model: {model_id}",
        f"Tool choice: {tool_choice}",
        f"Configured LiteLLM params: {json.dumps(litellm_params, indent=2, ensure_ascii=False, default=str)}",
    ]

    if assistant_message is None:
        lines.append("No assistant message in response.")
        return "\n".join(lines)

    # LiteLLM normalizes provider-specific reasoning into the standard
    # ``reasoning_content`` string when possible.  Fallback to other known
    # fields only when the standard field is absent.
    reasoning_content: Optional[str] = assistant_message.get("reasoning_content")
    if reasoning_content is not None:
        lines.append(f"reasoning_content ({len(reasoning_content)} chars):")
        lines.append(reasoning_content)
    else:
        # Provider-native fallbacks (rare after LiteLLM normalization).
        fallback_reasoning: Optional[Any] = (
            assistant_message.get("thinking_blocks")
            or assistant_message.get("reasoning")
            or assistant_message.get("reasoning_details")
        )
        if fallback_reasoning is not None:
            lines.append(f"reasoning (fallback, {type(fallback_reasoning).__name__}):")
            lines.append(json.dumps(fallback_reasoning, indent=2, ensure_ascii=False, default=str))
        else:
            lines.append("No reasoning fields present in assistant message.")
            lines.append(f"Available message keys: {list(assistant_message.keys())}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  SlowBurnAPIBackend
# ---------------------------------------------------------------------------


class SlowBurnAPIBackend(VLMBackend):
    """SlowBurn/LiteLLM backend — routes LLM calls to any supported provider.

    This class is a PURE TRANSPORT LAYER.  It:
        - Creates a SlowBurn LLM worker with cost/rate limits and retries.
        - Translates ``call()`` / ``call_batch()`` into
          ``self._llm.call_llm_batch()``.
        - Optionally wraps a ``response_type`` Typed class into a SlowBurn
          validator for automatic JSON parsing and schema validation.

    All calls — including single ``call()`` — go through
    ``call_llm_batch`` for uniform execution.  SlowBurn handles
    rate limiting, retries, and concurrent backpressure automatically.

    It has ZERO knowledge of distribution fitting, time series, proposals,
    kernels, priors, or any other domain concept.  All response schema
    enforcement is delegated to the ``response_type`` Typed class.

    DO NOT add domain-specific logic to this class.  If the VLM response
    needs domain-specific validation, add ``@field_validator`` methods to
    the ``response_type`` Typed class (in ``domains/*/prompts.py``), NOT here.
    """

    aliases: ClassVar[List[str]] = ["api"]

    litellm_model: str
    max_tokens: int
    temperature: float
    verbosity: int
    num_retries: int = 5
    retry_wait: Optional[float] = None
    retry_algorithm: str = "Linear"
    call_timeout: float = 180.0
    max_rpm: int = 0
    budget_usd: float = float("inf")
    budget_window: str = "hourly"
    api_key: Optional[str] = None
    litellm_params: Optional[Dict[str, Any]] = None
    model_cost: Optional[Dict[str, Any]] = None
    dataset_prefix: str = ""  # Prepend to log blocks for parallel execution tracking

    _llm: Any = PrivateAttr()

    def post_initialize(self) -> None:
        """Create the SlowBurn LLM worker.  No domain logic here."""
        if self.verbosity >= 3:
            # Enable LiteLLM's internal debug output for deep troubleshooting
            litellm.set_verbose = True
            logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
            logger.debug(
                f"[{self.dataset_prefix}] LiteLLM verbose logging ENABLED (verbosity={self.verbosity})"
            )

        if self.model_cost is not None:
            litellm.model_cost[self.litellm_model] = self.model_cost

        merged_litellm_params: Dict[str, Any] = {}
        if self.litellm_params is not None:
            merged_litellm_params.update(self.litellm_params)

        create_llm_kwargs: Dict[str, Any] = dict(
            model=self.litellm_model,
            budget_usd=self.budget_usd,
            window=self.budget_window,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.call_timeout,
            num_retries=self.num_retries,
            retry_wait=self.retry_wait,
            retry_algorithm=self.retry_algorithm,
            litellm_params=merged_litellm_params,
        )
        if self.api_key is not None:
            create_llm_kwargs["api_key"] = self.api_key
        if self.max_rpm > 0:
            create_llm_kwargs["max_rpm"] = self.max_rpm

        # Force GCRA from the PyMC side so behavior is stable across SlowBurn versions.
        # We pass the string name for compatibility with older/newer SlowBurn/concurry.
        create_llm_kwargs["rate_limit_algorithm"] = "GCRA"

        self._llm = create_llm(**create_llm_kwargs)
        if self.verbosity >= 1:
            rpm_desc: str = f", max_rpm={self.max_rpm}" if self.max_rpm > 0 else ""
            logger.debug(
                f"Ready: {self.litellm_model}{rpm_desc}, "
                f"timeout={self.call_timeout}s, "
                f"retry_wait={self.retry_wait:.1f}s, "
                f"rate_limit_algorithm=GCRA"
            )

    # -- Public API --------------------------------------------------------

    @validate
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
        """Send prompt + images + tools to the LLM.

        Delegates to ``call_batch()`` with a single-element list.

        Args:
            prompt: The text prompt to send to the LLM.
            images: Optional list of image paths to include in the request.
            tools: Optional list of tool schemas for function calling.
            response_type: Optional Typed class for automatic response parsing.
                When provided, the raw LLM response is parsed as JSON and
                validated against this schema.
            verbosity: Logging verbosity (0=silent, 1=progress, 2=detailed).
            dataset_prefix: Optional dataset marker for log identification.

        Returns:
            Raw response text when ``response_type`` is None, otherwise a
            validated ``response_type`` instance.

        Raises:
            ValueError: If the LLM response does not match ``response_type``.
        """
        results: List[Union[str, Typed]] = self.call_batch(
            prompts=[prompt],
            images_per_prompt=[images],
            response_type=response_type,
            verbosity=verbosity,
            tools=tools,
            dataset_prefix=dataset_prefix,
        )
        return results[0]

    @validate
    def call_batch(
        self,
        *,
        prompts: List[str],
        images_per_prompt: Optional[List[Optional[List[Union[str, Path]]]]] = None,
        response_type: Optional[Type[Typed]] = None,
        verbosity: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        dataset_prefix: str = "",
    ) -> List[Union[str, Typed]]:
        """Run multiple independent LLM calls concurrently via call_llm_batch.

        All calls share the same response_type, tools, and verbosity.
        SlowBurn handles rate limiting, retries, and backpressure.
        """
        effective_prefix: str = dataset_prefix if len(dataset_prefix) > 0 else self.dataset_prefix

        if len(prompts) == 0:
            return []

        if images_per_prompt is None:
            images_per_prompt = [None] * len(prompts)
        if len(images_per_prompt) != len(prompts):
            raise ValueError(
                f"images_per_prompt length ({len(images_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )

        resolved_images_per_prompt: List[Optional[List[Path]]] = []
        for imgs in images_per_prompt:
            resolved: List[Path] = _resolve_images(imgs)
            resolved_images_per_prompt.append(resolved if len(resolved) > 0 else None)

        if verbosity >= 1:
            tool_names_for_logging: List[str] = (
                [tool_dict["function"]["name"] for tool_dict in tools] if tools is not None else []
            )
            for prompt_index, prompt_text in enumerate(prompts):
                images_for_prompt: Optional[List[Path]] = resolved_images_per_prompt[prompt_index]
                image_paths_for_prompt: List[str] = (
                    [str(image_path) for image_path in images_for_prompt]
                    if images_for_prompt is not None
                    else []
                )
                logger.debug(
                    format_log_block(
                        title=f"[call_batch {prompt_index + 1}/{len(prompts)}] REQUEST",
                        body=(
                            f"Images ({len(image_paths_for_prompt)}): {image_paths_for_prompt}\n"
                            f"Tools: "
                            f"{tool_names_for_logging if len(tool_names_for_logging) > 0 else '(none)'}\n"
                            f"Prompt ({len(prompt_text)} chars):\n"
                            f"{prompt_text}"
                        ),
                        dataset_prefix=effective_prefix,
                    )
                )

        validator: Optional[Callable[[str], Typed]] = None
        if response_type is not None:
            validator = _build_typed_validator(
                response_type=response_type,
                verbosity=verbosity,
            )

        batch_kwargs: Dict[str, Any] = dict(
            prompts=prompts,
            images_per_prompt=resolved_images_per_prompt,
            image_detail="low",
            return_messages=True,
            validator=validator,
            verbosity=verbosity,
        )

        if tools is not None and len(tools) > 0:
            batch_kwargs["tools"]: Optional[List[Dict[str, Any]]] = tools
            batch_kwargs["tool_choice"]: Optional[str] = "auto"
        else:
            batch_kwargs["tools"]: Optional[List[Dict[str, Any]]] = None
            batch_kwargs["tool_choice"]: Optional[str] = None

        try:
            # The outer timeout must accommodate all retries (num_retries + 1
            # attempts, each with call_timeout seconds).
            total_timeout: float = self.call_timeout * (self.num_retries + 1)
            message_results: List[List[Dict[str, Any]]] = self._llm.call_llm_batch(
                **batch_kwargs,
            ).result(timeout=total_timeout)

            results: List[Union[str, Typed]] = []
            for response_index, messages in enumerate(message_results):
                assistant_message: Optional[Dict[str, Any]] = None
                for message in reversed(messages):
                    if message.get("role") == "assistant":
                        assistant_message = message
                        break
                if assistant_message is None:
                    raise ValueError(
                        f"No assistant message in call_batch response at index {response_index}. "
                        f"Messages: {messages}"
                    )

                logger.info(
                    format_log_block(
                        title=f"[call_batch {response_index + 1}/{len(message_results)}] REASONING TRACE",
                        body=_format_reasoning_trace(
                            model_id=self.litellm_model,
                            litellm_params=self.litellm_params,
                            tool_choice=batch_kwargs["tool_choice"],
                            assistant_message=assistant_message,
                        ),
                        dataset_prefix=effective_prefix,
                    )
                )

                response_text: Optional[str] = assistant_message.get("content")
                if response_text is None:
                    raise ValueError(
                        f"Assistant message in call_batch response at index {response_index} "
                        f"has no text content. Tool calls must be consumed through call_for_tool(). "
                        f"Message keys: {list(assistant_message.keys())}"
                    )

                if validator is not None:
                    results.append(validator(response_text))
                else:
                    results.append(response_text)

            if verbosity >= 1:
                for response_index, response_value in enumerate(results):
                    if isinstance(response_value, str):
                        response_len_chars: int = len(response_value)
                        response_kind: str = "raw text"
                    else:
                        response_len_chars = len(str(response_value))
                        response_kind = f"Typed<{type(response_value).__name__}>"
                    logger.debug(
                        format_log_block(
                            title=f"[call_batch {response_index + 1}/{len(results)}] RESPONSE",
                            body=(
                                f"Response kind: {response_kind}\n"
                                f"Response ({response_len_chars} chars):\n"
                                f"{response_value}"
                            ),
                            dataset_prefix=effective_prefix,
                        )
                    )

            return results

        except ValueError as exc:
            error_msg: str = format_exception_msg(exc)
            response_type_name: Optional[str] = response_type.__name__ if response_type is not None else None
            logger.error(
                f"VLM batch call failed ({len(prompts)} prompts).\n"
                f"  Response type: {response_type_name}\n"
                f"  Error: {error_msg}"
            )
            raise

    @validate
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

        Always routes through the standard SlowBurn ``call_llm_batch`` path
        (which uses ``litellm.acompletion``) so concurrency, rate limiting,
        and retries are preserved.
        """
        resolved_images: List[Path] = _resolve_images(images)

        if verbosity >= 1:
            tool_names: List[str] = [tool_dict["function"]["name"] for tool_dict in tools]
            image_paths: List[str] = [str(image_path) for image_path in resolved_images]
            logger.debug(
                format_log_block(
                    title="[call_for_tool] REQUEST",
                    body=(
                        f"Tool choice: {tool_choice}\n"
                        f"Tools: {tool_names}\n"
                        f"Images ({len(image_paths)}): {image_paths}\n"
                        f"Prompt ({len(prompt)} chars):\n"
                        f"{prompt}"
                    ),
                    dataset_prefix=dataset_prefix,
                )
            )

        try:
            # Outer timeout must accommodate all retry attempts.
            total_timeout: float = self.call_timeout * (self.num_retries + 1)
            batch_results: List[Any] = self._llm.call_llm_batch(
                prompts=[prompt],
                images_per_prompt=[resolved_images if len(resolved_images) > 0 else None],
                image_detail="low",
                return_messages=True,
                tools=tools,
                tool_choice=tool_choice,
                verbosity=verbosity,
            ).result(timeout=total_timeout)

            messages: List[Dict[str, Any]] = batch_results[0]

            assistant_message: Optional[Dict[str, Any]] = None
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    assistant_message = message
                    break

            if assistant_message is None:
                logger.warning("[call_for_tool] No assistant message in response")
                return ToolCallResponse()

            content: Optional[str] = assistant_message.get("content")
            raw_tool_calls: Optional[List[Dict[str, Any]]] = assistant_message.get("tool_calls")

            logger.info(
                format_log_block(
                    title="[call_for_tool] REASONING TRACE",
                    body=_format_reasoning_trace(
                        model_id=self.litellm_model,
                        litellm_params=self.litellm_params,
                        tool_choice=tool_choice,
                        assistant_message=assistant_message,
                    ),
                    dataset_prefix=dataset_prefix,
                )
            )

            parsed_tool_calls: List[ToolCallResult] = []
            if raw_tool_calls is not None:
                for tc in raw_tool_calls:
                    arguments_raw: str = tc["function"]["arguments"]
                    try:
                        arguments_parsed: Dict[str, Any] = (
                            json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
                        )
                    except (json.JSONDecodeError, TypeError):
                        logger.error(
                            f"[call_for_tool] Failed to parse tool call arguments for "
                            f"{tc['function']['name']}: {arguments_raw}. Using empty arguments."
                        )
                        arguments_parsed = {}
                    parsed_tool_calls.append(
                        ToolCallResult(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=arguments_parsed,
                        )
                    )

            tool_call_response: ToolCallResponse = ToolCallResponse(
                content=content,
                tool_calls=parsed_tool_calls,
            )

            if verbosity >= 1:
                content_chars: int = len(content) if content is not None else 0
                tool_call_names: List[str] = [tool_call.name for tool_call in parsed_tool_calls]
                logger.debug(
                    format_log_block(
                        title="[call_for_tool] RESPONSE",
                        body=(
                            f"Tool calls ({len(parsed_tool_calls)}): {tool_call_names}\n"
                            f"Assistant content ({content_chars} chars):\n"
                            f"{content}"
                        ),
                        dataset_prefix=dataset_prefix,
                    )
                )

            return tool_call_response

        except (ValueError, TimeoutError) as exc:
            error_msg: str = format_exception_msg(exc)
            tool_names_err: List[str] = [t["function"]["name"] for t in tools]
            logger.error(
                f"[call_for_tool] failed.\n"
                f"  Prompt ({len(prompt)} chars):\n"
                f"{prompt}\n"
                f"  Tools: {tool_names_err}\n"
                f"  Error: {error_msg}"
            )
            raise

    def stop(self) -> None:
        """Shut down the SlowBurn worker.  No domain logic here."""
        try:
            self._llm.stop()
        except (AttributeError, RuntimeError) as exc:
            logger.debug(f"SlowBurn stop encountered: {format_exception_msg(exc)}")

    def get_cost_report(self) -> str:
        """Return a markdown cost report.  No domain logic here."""
        try:
            reporter: Any = self._llm.get_reporter().result(timeout=5.0)
            report_text: str = reporter.to_markdown()
            return report_text
        except (AttributeError, RuntimeError, TimeoutError) as exc:
            logger.debug(f"Cost report unavailable: {format_exception_msg(exc)}")
            return "(cost report unavailable)"
