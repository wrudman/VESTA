import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import AzureOpenAI, OpenAI


def _required_env(name: str) -> str:
    if name not in os.environ:
        raise EnvironmentError(f"{name} must be set before running the Box Loop baseline.")
    value: str = os.environ[name]
    if len(value) == 0:
        raise EnvironmentError(f"{name} is set but empty. Provide a valid value before running.")
    return value


class LMExperimenter:
    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        max_tokens: int,
        throttle_llm_call: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self._throttle_llm_call: Optional[Callable[[], None]] = throttle_llm_call
        self.messages: List[Dict[str, Any]] = []
        self.all_messages: List[str] = []
        self.system: Optional[str] = None

        if model_name.startswith("azure/"):
            self._provider: str = "azure"
            self._api_model_name: str = model_name.removeprefix("azure/")
            self.llm: Any = AzureOpenAI(
                api_key=_required_env("AZURE_API_KEY"),
                api_version=_required_env("AZURE_API_VERSION"),
                azure_endpoint=_required_env("AZURE_API_BASE"),
            )
        elif model_name.startswith("openrouter/"):
            self._provider = "openrouter"
            self._api_model_name = model_name.removeprefix("openrouter/")
            self.llm = OpenAI(
                api_key=_required_env("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        elif "Qwen" in model_name or "together" in model_name:
            self._provider = "together"
            self._api_model_name = model_name
            self.llm = OpenAI(
                api_key=_required_env("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        else:
            raise ValueError(
                f"Model {model_name!r} is not supported by Box Loop. "
                f"Use 'azure/gpt-5.4-mini', "
                f"'openrouter/anthropic/claude-sonnet-4.6', or "
                f"'openrouter/moonshotai/kimi-k2.5'."
            )

    def set_system_message(self, message: str) -> None:
        self.all_messages.append(f"role:system, message:{message}")
        if self._provider == "azure":
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": message}],
                }
            )
        elif self._provider in ("openrouter", "together"):
            self.messages.append({"role": "system", "content": message})
        else:
            raise ValueError(f"Unknown provider {self._provider!r}.")

    def add_message(self, message: str, role: str = "user") -> None:
        self.all_messages.append(f"role:{role}, message:{message}")
        if self._provider == "azure":
            content_type: str = "output_text" if role == "assistant" else "input_text"
            self.messages.append(
                {
                    "role": role,
                    "content": [{"type": content_type, "text": message}],
                }
            )
        elif self._provider in ("openrouter", "together"):
            self.messages.append({"role": role, "content": message})
        else:
            raise ValueError(f"Unknown provider {self._provider!r}.")

    def _openrouter_extra_body(self) -> Dict[str, Any]:
        if "claude-sonnet-4.6" in self.model_name:
            return {"reasoning": {"max_tokens": 1024, "exclude": False}}
        elif "kimi-k2.5" in self.model_name:
            return {"reasoning": {"effort": "low", "exclude": False}}
        else:
            raise ValueError(
                f"Unsupported OpenRouter model {self.model_name!r}. "
                f"Only 'claude-sonnet-4.6' and 'kimi-k2.5' are configured."
            )

    def prompt_llm(self, request_prompt: str) -> str:
        self.add_message(request_prompt, role="user")
        if self._throttle_llm_call is not None:
            self._throttle_llm_call()

        if self._provider == "azure":
            response: Any = self.llm.responses.create(
                model=self._api_model_name,
                input=self.messages,
                max_output_tokens=self.max_tokens,
                reasoning={"effort": "low"},
            )
            full_response: str = response.output_text
        elif self._provider == "openrouter":
            response = self.llm.chat.completions.create(
                model=self._api_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self._openrouter_extra_body(),
            )
            full_response = response.choices[0].message.content
            if full_response is None:
                raise ValueError(f"OpenRouter returned no text content for model {self.model_name!r}.")
        elif self._provider == "together":
            response = self.llm.chat.completions.create(
                model=self._api_model_name,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            full_response = response.choices[0].message.content
            if full_response is None:
                raise ValueError(f"Together returned no text content for model {self.model_name!r}.")
        else:
            raise ValueError(f"Unknown provider {self._provider!r}.")

        self.add_message(full_response, role="assistant")
        return full_response

    def parse_response(self, response: str, is_observation: bool) -> Optional[str]:
        pattern: str = r"<observe>(.*?)</observe>" if is_observation else r"<answer>(.*?)</answer>"
        match: Optional[re.Match[str]] = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match is not None else None

    def prompt_llm_and_parse(
        self,
        request_prompt: str,
        is_observation: bool,
        max_tries: int = 4,
    ) -> Tuple[str, int]:
        used_retries: int = 0
        response: Optional[str] = None
        for _ in range(max_tries):
            full_response: str = self.prompt_llm(request_prompt)
            response = self.parse_response(full_response, is_observation)
            if response is not None:
                if len(re.findall(r"[0-9]+", response)) == 0:
                    response = None
            if response is None or "done" in response:
                if is_observation:
                    request_prompt = (
                        "Please stick to the specified format and respond using <observe> tags. "
                        "Continue making observations even if you think you have an accurate estimate. "
                        "Your previous response was not valid."
                    )
                else:
                    request_prompt = (
                        "Please stick to the specified format and respond using <answer> tags. "
                        "Make assumptions and provide your best guess. Your previous response was not valid."
                    )
                used_retries += 1
            else:
                break
        if used_retries == max_tries or response is None:
            raise ValueError("Failed to get a valid response after max retries.")
        return response, used_retries

    def generate_predictions(self, request_prompt: str) -> str:
        request_prompt += "\nAnswer in the following format:\n<answer>your answer</answer>."
        prediction: str
        used_retries: int
        prediction, used_retries = self.prompt_llm_and_parse(request_prompt, False)
        self.messages = self.messages[: -2 * (used_retries + 1)]
        return prediction

    def generate_actions(self, experiment_results: Optional[str] = None) -> str:
        if experiment_results is None:
            follow_up_prompt: str = (
                "Think about where to observe next. Articulate your strategy for choosing "
                "measurements in <thought>.\nProvide a new measurement point in the format:\n"
                "<thought>your thought</thought>\n<observe>your observation</observe>\n"
                "Make an observation now."
            )
        else:
            follow_up_prompt = (
                f"Result: {experiment_results}\nThink about where to observe next. "
                "Articulate your strategy for choosing measurements in <thought>.\n"
                "Provide a new measurement point in the format:\n"
                "<thought>your thought</thought>\n"
                "<observe>your observation (remember the type of inputs accepted)</observe>"
            )
        observe: str
        used_retries: int
        observe, used_retries = self.prompt_llm_and_parse(follow_up_prompt, True)
        return observe

    def print_log(self) -> None:
        for entry in self.all_messages:
            print(entry)
            print("------")
