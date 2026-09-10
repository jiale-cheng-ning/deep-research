import pytest

from deep_research.config import Settings
from deep_research.llm import BudgetExhaustedError, TokenBudget, TokenUsage


class TestTokenUsage:
    def test_default_zero(self):
        u = TokenUsage()
        assert u.total == 0

    def test_add(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=30)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 80
        assert c.total == 380


class TestTokenBudget:
    def test_remaining(self):
        b = TokenBudget(max_total=1000)
        assert b.remaining == 1000

    def test_record_and_remaining(self):
        b = TokenBudget(max_total=1000)
        b.record(TokenUsage(input_tokens=600, output_tokens=300))
        assert b.remaining == 100

    def test_check_raises_when_exhausted(self):
        b = TokenBudget(max_total=100)
        b.record(TokenUsage(input_tokens=100))
        with pytest.raises(BudgetExhaustedError):
            b.check()

    def test_check_passes_when_budget_remains(self):
        b = TokenBudget(max_total=1000)
        b.record(TokenUsage(input_tokens=100))
        b.check()  # should not raise


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.model_sonnet == "claude-sonnet-4-6"
        assert s.max_total_tokens == 500_000
        assert s.llm_concurrency == 3
        assert s.search_concurrency == 2
        assert s.deep_scrape_top_n == 3
        assert s.llm_max_retries == 3

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("DR_ANTHROPIC_API_KEY", "test-key-123")
        monkeypatch.setenv("DR_LLM_CONCURRENCY", "5")
        s = Settings()
        assert s.anthropic_api_key == "test-key-123"
        assert s.llm_concurrency == 5
