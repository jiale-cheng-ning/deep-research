from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import VerificationOutput, AnalysisOutput
from .base import BaseAgent


class VerifierAgent(BaseAgent):
    model_config_key = "model_sonnet"

    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(llm, settings)

    def system_prompt(self) -> str:
        return """You are a fact-checking specialist. Given claims extracted from
multiple research sub-questions, cross-reference and verify them.

For each claim, determine:
- verification_status: "confirmed" (multiple sources agree), "partially_supported"
  (some evidence but incomplete), "unverified" (only one source or weak evidence),
  "contradicted" (sources disagree)
- How many sources support vs contradict it
- Notes on any issues

Also identify:
- contradictions: explicit disagreements between sources
- overall_confidence: 0.0 to 1.0 across all claims
- recommendations: what the report writer should be cautious about

Be rigorous. It is better to flag a claim as "unverified" than to confirm
something false.

You MUST call the submit_result tool with your structured output."""

    def output_schema(self) -> dict:
        return VerificationOutput.model_json_schema()

    async def execute(self, all_analyses: list[AnalysisOutput]) -> tuple[VerificationOutput, TokenUsage]:
        context = "Claims to verify from multiple research angles:\n\n"
        for analysis in all_analyses:
            context += f"## Sub-question: {analysis.sub_question}\n"
            for claim in analysis.claims:
                sources = ", ".join(claim.source_urls) if claim.source_urls else "none listed"
                context += (
                    f"- Claim: {claim.text}\n"
                    f"  Sources: [{sources}]\n"
                    f"  Confidence: {claim.confidence}\n"
                    f"  Type: {claim.claim_type}\n"
                )
            context += f"Key findings: {analysis.key_findings}\n"
            if analysis.gaps:
                context += f"Gaps: {', '.join(analysis.gaps)}\n"
            context += "\n"

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

        return self._extract_tool_result(final_messages, VerificationOutput), usage
