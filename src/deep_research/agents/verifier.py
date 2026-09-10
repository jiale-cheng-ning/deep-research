import logging

from ..config import Settings
from ..llm import LLMClient, TokenUsage
from ..models import VerificationOutput, AnalysisOutput
from ..search import SearchTools
from .base import BaseAgent

logger = logging.getLogger("deep_research.agents.verifier")


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

If you are given ADDITIONAL EVIDENCE from independent verification searches,
use it to upgrade or downgrade the claim status accordingly.

Also identify:
- contradictions: explicit disagreements between sources
- overall_confidence: 0.0 to 1.0 across all claims
- recommendations: what the report writer should be cautious about

Be rigorous. It is better to flag a claim as "unverified" than to confirm
something false.

You MUST call the submit_result tool with your structured output."""

    def output_schema(self) -> dict:
        return VerificationOutput.model_json_schema()

    async def _gather_verification_evidence(
        self,
        all_analyses: list[AnalysisOutput],
        search_tools: SearchTools | None,
        max_claims: int = 5,
    ) -> str:
        """Run targeted searches for low-confidence / single-source claims."""
        if search_tools is None:
            return ""

        # Collect claims that need external verification
        candidates = []
        for analysis in all_analyses:
            for claim in analysis.claims:
                needs_check = (
                    claim.confidence < 0.7
                    or len(claim.source_urls) <= 1
                    or claim.claim_type in ("statistic", "prediction")
                )
                if needs_check:
                    candidates.append(claim)

        # Prioritize: lowest confidence first, cap the number of searches
        candidates.sort(key=lambda c: c.confidence)
        to_verify = candidates[:max_claims]

        if not to_verify:
            return ""

        logger.info("Verifier running %d targeted verification searches", len(to_verify))
        evidence_parts = ["## Independent Verification Search Results\n"]

        for i, claim in enumerate(to_verify, 1):
            query = claim.text[:150]
            try:
                results = await search_tools.search(query, source="tavily")
            except Exception as e:
                logger.warning("Verification search failed for claim %d: %s", i, e)
                continue

            if not results:
                continue

            evidence_parts.append(f"\n### Verification for claim: {claim.text[:120]}\n")
            for r in results[:3]:
                snippet = (r.get("content") or "")[:400]
                evidence_parts.append(f"- [{r.get('title', '')}]({r['url']})\n  {snippet}\n")

        text = "".join(evidence_parts)
        return text if len(evidence_parts) > 1 else ""

    async def execute(
        self,
        all_analyses: list[AnalysisOutput],
        search_tools: SearchTools | None = None,
    ) -> tuple[VerificationOutput, TokenUsage]:
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

        # Add independent evidence from targeted verification searches
        evidence = await self._gather_verification_evidence(all_analyses, search_tools)
        if evidence:
            context += "\n" + evidence

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
