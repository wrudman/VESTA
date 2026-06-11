"""vLLM offline-engine backend and Concurry Ray worker for Qwen3-VL models.

The ``VLLMQwenBackend`` class wraps a vLLM ``LLM`` engine with XML-based
tool calling.  ``VLLMModelWorker`` is a Ray actor that holds the
GPU-resident model and exposes an inference method for DatasetFitter
workers to call remotely.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from concurry import worker
from morphic import Typed, validate
from morphic.string import format_exception_msg
from pydantic import PrivateAttr

from .base import ToolCallResponse, VLMBackend
from .parsing import parse_json_from_text

logger: logging.Logger = logging.getLogger("vlm_backends.vllm")


def _parse_tool_calls(raw_text: str) -> List[Dict[str, Any]]:
    """Extract ``<tool_call>`` blocks from raw Qwen generation output.

    Qwen models emit tool calls as XML::

        <tool_call>
        {"name": "fn", "arguments": {"k": "v"}}
        </tool_call>

    Returns a list of dicts with keys ``name`` and ``arguments``.
    """
    pattern: re.Pattern[str] = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    results: List[Dict[str, Any]] = []
    for match in pattern.finditer(raw_text):
        match: re.Match[str]
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return results


def _strip_thinking_and_tool_calls(raw_text: str) -> str:
    """Remove ``<think>`` and ``<tool_call>`` blocks, leaving the JSON answer."""
    text: str = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return text.strip()


class VLLMQwenBackend(VLMBackend):
    """VLM backend using the vLLM offline ``LLM`` class on a local GPU.

    Loads the model once at construction time.  Subsequent ``call()``
    invocations run inference without any network round-trip.  Tool schemas
    are injected via ``apply_chat_template(tools=...)`` and the model emits
    XML ``<tool_call>`` blocks which are parsed client-side.
    """

    aliases: ClassVar[List[str]] = ["vllm_qwen", "vllm"]

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enable_thinking: bool = True
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    _llm_engine: Any = PrivateAttr()
    _processor: Any = PrivateAttr()

    def post_initialize(self) -> None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if "CC" not in os.environ:
            os.environ["CC"] = "gcc"
        if "CXX" not in os.environ:
            os.environ["CXX"] = "g++"

        from transformers import AutoProcessor
        from vllm import LLM

        logger.info("Loading vLLM engine: %s ...", self.model_name)
        self._llm_engine = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=True,
            hf_overrides={
                "text_config": {"tie_word_embeddings": False},
            },
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("Ready: %s", self.model_name)

    @validate
    def call(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_type: Optional[Type[Typed]] = None,
        verbosity: int,
    ) -> Union[str, Typed]:
        """Run one VLM inference call on the local GPU."""
        results: List[Union[str, Typed]] = self.call_batch(
            prompts=[prompt],
            images_per_prompt=[images],
            response_type=response_type,
            verbosity=verbosity,
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
    ) -> List[Union[str, Typed]]:
        """Run multiple VLM inference calls on the local GPU (sequential).

        vLLM's ``generate()`` batches internally when given multiple
        prompts, but the image handling and response parsing here is
        per-prompt, so we iterate sequentially.
        """
        if images_per_prompt is None:
            images_per_prompt = [None] * len(prompts)
        if len(images_per_prompt) != len(prompts):
            raise ValueError(
                f"images_per_prompt length ({len(images_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )

        results: List[Union[str, Typed]] = []
        for prompt, images in zip(prompts, images_per_prompt):
            result: Union[str, Typed] = self._call_single(
                prompt=prompt,
                images=images,
                response_type=response_type,
                verbosity=verbosity,
            )
            results.append(result)
        return results

    def _call_single(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_type: Optional[Type[Typed]] = None,
        verbosity: int,
    ) -> Union[str, Typed]:
        """Run one VLM inference call on the local GPU (internal)."""
        from vllm import SamplingParams

        if images is None or len(images) == 0:
            raise ValueError("VLLMQwenBackend requires at least one image. Pass images=[path_to_image].")
        image_path: Path = Path(images[0])

        with open(image_path, "rb") as f:
            image_b64: str = base64.b64encode(f.read()).decode("utf-8")

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        apply_kwargs: Dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        if tools is not None and len(tools) > 0:
            apply_kwargs["tools"] = tools

        formatted_prompt: str = self._processor.apply_chat_template(messages, **apply_kwargs)

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        outputs = self._llm_engine.generate(
            [formatted_prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        raw_text: str = outputs[0].outputs[0].text
        if verbosity >= 2:
            logger.debug("Raw response: %s", raw_text)

        if response_type is None:
            return raw_text

        tool_calls: List[Dict[str, Any]] = _parse_tool_calls(raw_text)
        clean_text: str = _strip_thinking_and_tool_calls(raw_text)

        try:
            parsed: Dict[str, Any] = parse_json_from_text(clean_text)
        except (ValueError, json.JSONDecodeError) as exc:
            if len(tool_calls) > 0:
                tool_names: List[str] = [tc["name"] for tc in tool_calls]
                raise ValueError(
                    f"VLM returned tool call(s) {tool_names} but no JSON body "
                    f"with model proposals. Parse error: {format_exception_msg(exc)}"
                ) from exc
            raise ValueError(f"Failed to parse JSON from VLM response: {format_exception_msg(exc)}") from exc

        if len(tool_calls) > 0:
            first_tool_call: Dict[str, Any] = tool_calls[0]
            parsed["selected_tool"] = first_tool_call["name"]
            parsed["selected_tool_args"] = first_tool_call["arguments"]

        try:
            result: Typed = response_type(**parsed)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"VLM response does not match {response_type.__name__} schema: "
                f"{format_exception_msg(exc)}. "
                f"Parsed keys: {list(parsed.keys())}"
            ) from exc

        return result

    @validate
    def call_for_tool(
        self,
        *,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        tools: List[Dict[str, Any]],
        tool_choice: str,
        verbosity: int,
    ) -> ToolCallResponse:
        """vLLM tool-calling uses XML-based tool calls embedded in the response.

        Not yet implemented as a separate method — the old ``call()`` path
        handled tool parsing inline.  Raise to surface the gap clearly.
        """
        raise NotImplementedError(
            "VLLMQwenBackend.call_for_tool() is not implemented. "
            "Use SlowBurnAPIBackend for the agentic tool loop."
        )

    def stop(self) -> None:
        """vLLM engine cleanup (no-op; engine is process-local)."""
        pass

    def get_cost_report(self) -> str:
        """Local GPU inference has no API cost."""
        return "(local vLLM — no API cost)"


@worker(mode="ray")
class VLLMModelWorker:
    """Ray actor holding a vLLM engine on GPU for Qwen3-VL inference.

    Internally delegates to ``VLLMQwenBackend``.  Exposes
    ``generate_feedback()`` which DatasetFitter workers call to get VLM
    predictions with optional function-calling tool selection.
    """

    def __init__(
        self,
        *,
        model_name: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enable_thinking: bool,
    ):
        self._backend: VLLMQwenBackend = VLLMQwenBackend(
            model_name=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_thinking=enable_thinking,
        )

    # @validate omitted: Worker methods are serialized for Ray remote execution.
    # validate_call closure is not picklable across process boundaries.
    def generate_feedback(
        self,
        *,
        prompt: str,
        image_path: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Typed]:
        """Run one VLM inference call and return parsed result or raw text."""
        return self._backend.call(
            prompt=prompt,
            images=[image_path],
            tools=tools,
            verbosity=1,
        )
