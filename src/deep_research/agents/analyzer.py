from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import AnalysisOutput, SearchOutput
from .base import BaseAgent


class AnalyzerAgent(BaseAgent):
    model_config_key = "model_sonnet"

    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(llm, settings)

    def system_prompt(self) -> str:
        return """You are a research analyst. Given search results for a sub-question,
extract all factual claims and key information.

For each claim:
- State it precisely and concisely
- Cite which source URL(s) support it
- Rate your confidence (0.0 to 1.0)
- Classify as: "fact", "opinion", "statistic", or "prediction"

Also provide:
- key_findings: a summary paragraph of the most important discoveries
- gaps: what information was NOT found or is missing

Be thorough. Extract ALL notable claims, not just the most obvious ones.

You MUST call the submit_result tool with your structured output."""

    def output_schema(self) -> dict:
        return AnalysisOutput.model_json_schema()

    async def execute(self, search_output: SearchOutput) -> tuple[AnalysisOutput, TokenUsage]:
        context = f"Sub-question: {search_output.sub_question}\n\nSearch results:\n"
        for i, r in enumerate(search_output.results):
            content = r.full_content or r.snippet
            context += f"\n[{i+1}] {r.title}\nURL: {r.url}\n{content}\n"

        messages = [{"role": "user", "content": context}]
        tool = self._make_structured_tool()

        final_messages, usage = await self.llm.run_agent(
            model=self.model,
            system=self.system_prompt(),
            messages=messages,
            tools=[tool],
            max_output_tokens=8192,
            force_tool=True,
        )

        output = self._extract_tool_result(final_messages, AnalysisOutput)
        output.sub_question = search_output.sub_question
        return output, usage
