from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import ResearchPlan
from .base import BaseAgent


class OrchestratorAgent(BaseAgent):
    model_config_key = "model_sonnet"

    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(llm, settings)

    def system_prompt(self) -> str:
        return """You are a research planning expert. Given a user's question,
decompose it into 3-5 specific sub-questions that together will answer the original
question comprehensively.

For each sub-question, specify:
- The exact search query to use (make it a good search engine query)
- Which search source is preferred (tavily for quick overviews, firecrawl for deep
  page content, both for comprehensive coverage)
- Priority (1=highest, 5=lowest)

Think about: What angles are missing? What needs recent data? What needs academic
sources vs. news vs. technical docs?

You MUST call the submit_result tool with your research plan."""

    def output_schema(self) -> dict:
        return ResearchPlan.model_json_schema()

    async def execute(self, user_question: str) -> tuple[ResearchPlan, TokenUsage]:
        messages = [{"role": "user", "content": f"Research question: {user_question}"}]
        tool = self._make_structured_tool()

        final_messages, usage = await self.llm.run_agent(
            model=self.model,
            system=self.system_prompt(),
            messages=messages,
            tools=[tool],
            max_output_tokens=4096,
            force_tool=True,
        )

        plan = self._extract_tool_result(final_messages, ResearchPlan)
        plan.original_question = user_question
        return plan, usage
