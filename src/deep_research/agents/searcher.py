import logging

from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import SearchOutput, SubQuestion
from ..search import SearchTools
from .base import BaseAgent

logger = logging.getLogger("deep_research.agents.searcher")


class SearcherAgent(BaseAgent):
    model_config_key = "model_haiku"

    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(llm, settings)

    def system_prompt(self) -> str:
        return """You are a web research assistant. You are given search results for
a research sub-question. Evaluate each result's relevance to the sub-question.

For each result, provide:
- title, url, snippet (a concise summary of the relevant content)
- relevance_score: 0.0 to 1.0 (how relevant is this to the sub-question)
- source: which search engine it came from

Only include results with relevance_score >= 0.3. Sort by relevance_score descending.

You MUST call the submit_result tool with your structured output."""

    def output_schema(self) -> dict:
        return SearchOutput.model_json_schema()

    async def execute(
        self, sub_question: SubQuestion, search_tools: SearchTools
    ) -> tuple[SearchOutput, TokenUsage]:
        source = sub_question.preferred_source.value
        raw_results = await search_tools.search(sub_question.question, source=source)

        if not raw_results:
            empty = SearchOutput(
                sub_question=sub_question.question,
                results=[],
                total_results_found=0,
            )
            return empty, TokenUsage()

        # Deep-scrape top-N results for richer analyzer context
        top_n = getattr(self.settings, "deep_scrape_top_n", 3) if self.settings else 3
        if top_n > 0:
            raw_results = await search_tools.deep_scrape_top(raw_results, top_n)
            scraped_count = sum(1 for r in raw_results if r.get("scraped"))
            logger.info(
                "Deep-scraped %d/%d pages for '%s'",
                scraped_count, top_n, sub_question.question[:60],
            )

        context = f"Sub-question: {sub_question.question}\n\nSearch results:\n"
        for i, r in enumerate(raw_results):
            content_preview = r["content"][:2000] if r["content"] else r.get("title", "")
            scraped_flag = " (full page)" if r.get("scraped") else ""
            context += (
                f"\n--- Result {i+1}{scraped_flag} ---\n"
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Source: {r.get('source', 'unknown')}\n"
                f"Content: {content_preview}\n"
            )

        messages = [{"role": "user", "content": context}]
        tool = self._make_structured_tool()

        final_messages, usage = await self.llm.run_agent(
            model=self.model,
            system=self.system_prompt(),
            messages=messages,
            tools=[tool],
            max_output_tokens=4096,
            force_tool=True,
        )

        output = self._extract_tool_result(final_messages, SearchOutput)
        output.sub_question = sub_question.question
        return output, usage
