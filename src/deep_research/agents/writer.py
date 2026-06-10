from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import (
    FinalReport, AnalysisOutput, SearchOutput, VerificationOutput,
)
from .base import BaseAgent


class WriterAgent(BaseAgent):
    model_config_key = "model_opus"

    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(llm, settings)

    def system_prompt(self) -> str:
        return """You are an expert research report writer. Synthesize the verified
research findings into a comprehensive, well-structured report.

Requirements:
- Use numbered citations [1], [2] etc. throughout the text
- Each citation number must correspond to a real source URL in the citations list
- Include an executive summary (2-3 paragraphs)
- Structure with clear headings and sub-sections
- Note confidence levels ("high", "medium", "low") for each section
- Include a methodology note explaining the multi-agent research process
- Write in the same language as the original question
- Be thorough but readable — aim for 2000-4000 words

Structure:
- Title
- Executive Summary
- Multiple themed sections (each with heading, content, confidence)
- Conclusions
- Citations list
- Methodology note

You MUST call the submit_result tool with your structured report."""

    def output_schema(self) -> dict:
        return FinalReport.model_json_schema()

    async def execute(
        self,
        original_question: str,
        verification: VerificationOutput,
        all_analyses: list[AnalysisOutput],
        all_search_outputs: list[SearchOutput],
    ) -> tuple[FinalReport, TokenUsage]:
        context = self._build_context(
            original_question, verification, all_analyses, all_search_outputs
        )

        messages = [{"role": "user", "content": context}]
        tool = self._make_structured_tool()

        final_messages, usage = await self.llm.run_agent(
            model=self.model,
            system=self.system_prompt(),
            messages=messages,
            tools=[tool],
            max_output_tokens=16384,
            force_tool=True,
        )

        return self._extract_tool_result(final_messages, FinalReport), usage

    def _build_context(
        self,
        question: str,
        verification: VerificationOutput,
        analyses: list[AnalysisOutput],
        searches: list[SearchOutput],
    ) -> str:
        parts = [f"Original research question: {question}\n"]

        parts.append("## Verification Results\n")
        parts.append(f"Overall confidence: {verification.overall_confidence}\n")
        if verification.contradictions:
            parts.append("Contradictions found:\n")
            for c in verification.contradictions:
                parts.append(f"- {c}\n")
        parts.append(f"Recommendations: {verification.recommendations}\n\n")

        parts.append("Verified claims:\n")
        for vc in verification.verified_claims:
            parts.append(
                f"- [{vc.verification_status}] {vc.claim_text}\n"
                f"  Supporting: {vc.supporting_sources}, Contradicting: {vc.contradicting_sources}\n"
                f"  Notes: {vc.notes}\n"
            )

        parts.append("\n## Detailed Analysis\n")
        for analysis in analyses:
            parts.append(f"### {analysis.sub_question}\n")
            parts.append(f"Key findings: {analysis.key_findings}\n")
            for claim in analysis.claims:
                parts.append(f"- {claim.text} (confidence: {claim.confidence})\n")
            if analysis.gaps:
                parts.append(f"Gaps: {', '.join(analysis.gaps)}\n")
            parts.append("\n")

        parts.append("## Available Sources for Citation\n")
        citation_num = 1
        seen_urls: set[str] = set()
        for search_out in searches:
            for r in search_out.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    content_preview = (r.full_content or r.snippet)[:500]
                    parts.append(f"[{citation_num}] {r.title}\nURL: {r.url}\n{content_preview}\n\n")
                    citation_num += 1

        return "".join(parts)
