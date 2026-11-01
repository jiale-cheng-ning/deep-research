import anthropic
import asyncio
import json
import logging
import random
from dataclasses import dataclass, field

from .config import Settings

logger = logging.getLogger("deep_research.llm")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class _RetryableAPIError(Exception):
    """Wrap API errors that should be retried."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )


@dataclass
class TokenBudget:
    max_total: int = 500_000
    usage: TokenUsage = field(default_factory=TokenUsage)

    def record(self, usage: TokenUsage) -> None:
        self.usage = self.usage + usage

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self.usage.total)

    def check(self) -> None:
        if self.remaining <= 0:
            raise BudgetExhaustedError(
                f"Token budget exhausted: {self.usage.total:,}/{self.max_total:,}"
            )


class BudgetExhaustedError(Exception):
    pass


class LLMClient:
    """Async Anthropic client with agentic tool-calling loop, retries, and budget tracking."""

    def __init__(self, settings: Settings, budget: TokenBudget):
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or "not-needed",
            base_url=settings.anthropic_base_url,
            default_headers={
                "Authorization": f"Bearer {settings.anthropic_api_key}",
            },
        )
        self.budget = budget
        self.max_retries = settings.llm_max_retries
        self.retry_base_delay = settings.llm_retry_base_delay
        if settings.llm_concurrency < 1:
            raise ValueError("llm_concurrency must be at least 1")
        self.semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def _create_with_retry(self, **kwargs):
        """Call messages.create with exponential backoff on retryable errors."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    self.budget.check()
                    response = await self.client.messages.create(**kwargs)
                    # Account before releasing the slot, so queued requests see it.
                    self.budget.record(TokenUsage(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                        cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                    ))
                    return response
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None) or 0
                if status not in _RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise
                last_exc = e
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                if attempt >= self.max_retries:
                    raise
                last_exc = e
            except Exception as e:
                # Unknown error — do not retry
                raise

            delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, self.max_retries + 1, last_exc, delay,
            )
            await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover

    async def run_agent(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
        tool_executor=None,
        force_tool: bool = False,
    ) -> tuple[list[dict], TokenUsage]:
        if tool_executor is None:
            tool_executor = self._default_tool_executor

        total_usage = TokenUsage()
        local_messages = list(messages)

        # Pin tool_choice to the exact structured-output tool name
        if force_tool and tools:
            tool_choice = {"type": "tool", "name": tools[0]["name"]}
        else:
            tool_choice = anthropic.NOT_GIVEN

        for turn in range(10):
            self.budget.check()

            response = await self._create_with_retry(
                model=model,
                max_tokens=max_output_tokens,
                system=system,
                messages=local_messages,
                tools=tools or [],
                tool_choice=tool_choice,
                temperature=temperature,
            )

            turn_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            )
            total_usage = total_usage + turn_usage

            # Append assistant response
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            local_messages.append({"role": "assistant", "content": assistant_content})

            # submit_result is terminal output, rather than an action to repeat.
            if force_tool and tools and any(
                b.get("type") == "tool_use" and b.get("name") == tools[0]["name"]
                for b in assistant_content
            ):
                break

            # If not a tool_use stop, we're done
            if response.stop_reason != "tool_use":
                break

            # Execute each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = await tool_executor(block.name, block.input)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result if isinstance(result, str) else json.dumps(result),
                    })

            local_messages.append({"role": "user", "content": tool_results})

        return local_messages, total_usage

    @staticmethod
    async def _default_tool_executor(name: str, input_data: dict) -> str:
        """Default executor: just return the input as JSON (used for structured output)."""
        return json.dumps(input_data, ensure_ascii=False)
