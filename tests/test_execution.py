import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from deep_research.config import Settings
from deep_research.llm import LLMClient, TokenBudget, TokenUsage
from deep_research.models import FinalReport, ResearchPlan, VerificationOutput
from deep_research.pipeline import ResearchPipeline


def client(monkeypatch, concurrency=2):
    sdk = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    monkeypatch.setattr("deep_research.llm.anthropic.AsyncAnthropic", Mock(return_value=sdk))
    return LLMClient(Settings(llm_concurrency=concurrency), TokenBudget()), sdk


def response():
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        content=[SimpleNamespace(type="tool_use", id="result", name="submit_result", input={"ok": True})],
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_structured_result_stops_after_one_paid_request(monkeypatch):
    llm, sdk = client(monkeypatch)
    sdk.messages.create.return_value = response()
    messages, usage = await llm.run_agent(
        model="mock", system="mock", messages=[], tools=[{"name": "submit_result"}],
        force_tool=True,
    )
    assert sdk.messages.create.await_count == 1
    assert usage.total == llm.budget.usage.total == 15
    assert messages[-1]["content"][0]["input"] == {"ok": True}


@pytest.mark.asyncio
async def test_concurrent_agents_share_request_limit(monkeypatch):
    llm, sdk = client(monkeypatch)
    active = peak = 0

    async def create(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response()

    sdk.messages.create.side_effect = create
    await asyncio.gather(*[
        llm.run_agent(model="mock", system="mock", messages=[],
                      tools=[{"name": "submit_result"}], force_tool=True)
        for _ in range(7)
    ])
    assert peak == 2
    assert llm.budget.usage.total == 105


def pipeline(monkeypatch, tmp_path, run_id="resume"):
    monkeypatch.setattr("deep_research.pipeline.LLMClient", Mock())
    return ResearchPipeline(Settings(
        anthropic_api_key="test", tavily_api_key="test", checkpoint_dir=str(tmp_path),
    ), run_id=run_id)


@pytest.mark.asyncio
async def test_interrupted_pipeline_resumes_without_repeating_completed_stages(monkeypatch, tmp_path):
    first = pipeline(monkeypatch, tmp_path)
    plan = ResearchPlan(sub_questions=[], reasoning="test")

    async def planning(question):
        first.budget.record(TokenUsage(input_tokens=12))
        return plan, TokenUsage(input_tokens=12)

    first.orchestrator.execute = AsyncMock(side_effect=planning)
    first._run_searches = AsyncMock(return_value=([], TokenUsage()))
    first._run_analyses = AsyncMock(side_effect=RuntimeError("interrupted"))
    with pytest.raises(RuntimeError, match="interrupted"):
        await first.run("question")

    second = pipeline(monkeypatch, tmp_path)
    second.orchestrator.execute = AsyncMock(side_effect=AssertionError("must resume"))
    second._run_searches = AsyncMock(side_effect=AssertionError("must resume"))
    second._run_analyses = AsyncMock(return_value=([], TokenUsage()))
    second._run_verification = AsyncMock(return_value=(
        VerificationOutput(verified_claims=[], overall_confidence=0.5), TokenUsage(),
    ))
    report = FinalReport(title="done", executive_summary="summary", sections=[],
                         conclusions="done", citations=[])
    second.writer.execute = AsyncMock(return_value=(report, TokenUsage()))
    assert (await second.run("question")).title == "done"
    assert second.budget.usage.total == 12
    second.orchestrator.execute.assert_not_awaited()
    second._run_searches.assert_not_awaited()
    third = pipeline(monkeypatch, tmp_path)
    third.orchestrator.execute = AsyncMock(side_effect=AssertionError("must resume"))
    third.writer.execute = AsyncMock(side_effect=AssertionError("must resume report"))
    assert (await third.run("question")).title == "done"
    assert third.budget.usage.total == 12
    third.writer.execute.assert_not_awaited()
    with pytest.raises(ValueError, match="different question"):
        await second.run("another question")


@pytest.mark.asyncio
async def test_invalid_checkpoint_rebuilds_downstream(monkeypatch, tmp_path):
    p = pipeline(monkeypatch, tmp_path)
    p._question = "q"
    p._checkpoint("a", {"format": 1, "question": "q", "data": "invalid", "usage": {}})
    p._checkpoint("b", {"format": 1, "question": "q", "data": [99], "usage": {}})
    first = AsyncMock(return_value=([1], TokenUsage()))
    second = AsyncMock(return_value=([2], TokenUsage()))
    assert (await p._stage("a", first, list[int]))[0] == [1]
    assert (await p._stage("b", second, list[int]))[0] == [2]
    second.assert_awaited_once()
