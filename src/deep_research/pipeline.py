import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from pydantic import TypeAdapter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .llm import LLMClient, TokenBudget, TokenUsage, BudgetExhaustedError
from .models import (
    FinalReport, ResearchPlan, SearchOutput, AnalysisOutput,
    VerificationOutput, VerifiedClaim, SubQuestion, SearchSource,
)
from .search import SearchTools
from .agents.orchestrator import OrchestratorAgent
from .agents.searcher import SearcherAgent
from .agents.analyzer import AnalyzerAgent
from .agents.verifier import VerifierAgent
from .agents.writer import WriterAgent

logger = logging.getLogger("deep_research.pipeline")


class ResearchPipeline:
    def __init__(self, settings: Settings, run_id: str | None = None):
        self.settings = settings
        self.budget = TokenBudget(max_total=settings.max_total_tokens)
        self.llm = LLMClient(settings, self.budget)
        self.search_tools = SearchTools(settings)
        self.console = Console()

        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.checkpoint_dir = Path(settings.checkpoint_dir) / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.orchestrator = OrchestratorAgent(self.llm, settings)
        self.searcher = SearcherAgent(self.llm, settings)
        self.analyzer = AnalyzerAgent(self.llm, settings)
        self.verifier = VerifierAgent(self.llm, settings)
        self.writer = WriterAgent(self.llm, settings)
        self._resume_valid = True

    # ------------------------------------------------------------------ utils
    def _checkpoint(self, name: str, data) -> None:
        """Persist intermediate artifacts for resume / debugging."""
        path = self.checkpoint_dir / f"{name}.json"
        try:
            payload = data.model_dump() if hasattr(data, "model_dump") else data
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            logger.debug("Checkpoint saved: %s", path)
        except Exception as e:
            logger.warning("Failed to save checkpoint %s: %s", name, e)

    def _load_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to load checkpoint %s: %s", name, e)
        return None

    async def _stage(self, name, operation, output_type):
        adapter = TypeAdapter(output_type)
        saved = self._load_checkpoint(name) if self._resume_valid else None
        if saved is not None:
            try:
                if saved["question"] != self._question or saved["format"] != 1:
                    raise ValueError("Checkpoint belongs to a different question or format")
                output = adapter.validate_python(saved["data"])
                usage = TokenUsage(**saved["usage"])
                if any(value < 0 for value in asdict(usage).values()):
                    raise ValueError("Invalid checkpoint usage")
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid checkpoint %s: %s", name, exc)
            else:
                self.budget.record(usage)
                self.console.print(f"  [cyan]Resumed {name}[/cyan]")
                return output, usage
        # Rebuilding an upstream stage invalidates all downstream checkpoints.
        self._resume_valid = False
        before = asdict(self.budget.usage)
        output, _ = await operation()
        usage = TokenUsage(**{
            key: value - before[key] for key, value in asdict(self.budget.usage).items()
        })
        self._checkpoint(name, {
            "format": 1, "question": self._question,
            "data": adapter.dump_python(output, mode="json"), "usage": asdict(usage),
        })
        return output, usage

    def _budget_ok(self, min_tokens: int = 5_000) -> bool:
        return self.budget.remaining >= min_tokens

    def _show_plan(self, plan: ResearchPlan):
        table = Table(title="Research Plan", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Sub-Question", style="cyan", min_width=40)
        table.add_column("Source", style="green", width=10)
        table.add_column("Priority", style="yellow", width=8)
        for i, sq in enumerate(plan.sub_questions, 1):
            table.add_row(str(i), sq.question, sq.preferred_source.value, str(sq.priority))
        self.console.print(table)
        self.console.print(f"[dim]Reasoning: {plan.reasoning}[/dim]\n")

    def _show_usage(self, label: str, usage: TokenUsage):
        self.console.print(
            f"  [dim]{label} tokens: "
            f"in={usage.input_tokens:,} out={usage.output_tokens:,} "
            f"total={usage.total:,}[/dim]"
        )

    def _show_budget_summary(self, total_usage: TokenUsage):
        table = Table(title="Token Budget Summary", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total Input Tokens", f"{total_usage.input_tokens:,}")
        table.add_row("Total Output Tokens", f"{total_usage.output_tokens:,}")
        table.add_row("Grand Total", f"[bold]{total_usage.total:,}[/bold]")
        table.add_row("Budget Remaining", f"{self.budget.remaining:,}")
        pct = self.budget.usage.total / self.budget.max_total * 100 if self.budget.max_total else 0
        table.add_row("Budget Used", f"{pct:.1f}%")
        table.add_row("Run ID", self.run_id)
        table.add_row("Checkpoints", str(self.checkpoint_dir))
        self.console.print(table)

    def _validate_settings(self) -> None:
        if not self.settings.anthropic_api_key:
            raise ValueError(
                "DR_ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        if not self.settings.tavily_api_key and not self.settings.firecrawl_api_key:
            raise ValueError(
                "At least one of DR_TAVILY_API_KEY / DR_FIRECRAWL_API_KEY must be set."
            )

    # ------------------------------------------------------------------ pipeline
    async def run(self, question: str) -> FinalReport:
        self._validate_settings()
        self.budget.usage = TokenUsage()
        self._resume_valid = True
        self._question = question
        # Never combine cached stages from different research questions/settings.
        identity = {"question": question, "settings": self.settings.model_dump(
            exclude={"anthropic_api_key", "tavily_api_key", "firecrawl_api_key",
                     "max_total_tokens", "verbose", "checkpoint_dir"}
        )}
        previous = self._load_checkpoint("00_run")
        if previous is not None and previous != identity:
            raise ValueError("Run ID belongs to a different question or settings; use a new --run-id")
        if previous is None:
            self._resume_valid = False
            self._checkpoint("00_run", identity)
        usage = TokenUsage()

        # ---- Stage 1: Plan
        self.console.print(Panel("[bold cyan]Stage 1: Planning Research Strategy[/bold cyan]"))
        plan, plan_usage = await self._stage(
            "01_plan", lambda: self.orchestrator.execute(question), ResearchPlan
        )
        usage = usage + plan_usage
        self._show_plan(plan)
        self._show_usage("Orchestrator", plan_usage)

        # ---- Stage 2: Parallel Search
        self.console.print(Panel("[bold yellow]Stage 2: Searching (parallel)[/bold yellow]"))
        search_outputs, search_usage = await self._stage(
            "02_search", lambda: self._run_searches(plan), list[SearchOutput]
        )
        usage = usage + search_usage
        self._show_usage("Searchers", search_usage)

        # ---- Stage 3: Parallel Analysis
        self.console.print(Panel("[bold green]Stage 3: Analyzing Claims (parallel)[/bold green]"))
        analysis_outputs, analysis_usage = await self._stage(
            "03_analysis", lambda: self._run_analyses(search_outputs), list[AnalysisOutput]
        )
        usage = usage + analysis_usage
        self._show_usage("Analyzers", analysis_usage)

        # ---- Stage 3.5: Second-round research for identified gaps (optional)
        if self._budget_ok(min_tokens=20_000) and plan.estimated_rounds > 1:
            analysis_outputs, extra_usage = await self._stage(
                "03b_gapfill", lambda: self._run_gap_fill(analysis_outputs, search_outputs),
                list[AnalysisOutput],
            )
            usage = usage + extra_usage
        elif self._load_checkpoint("03b_gapfill") is not None:
            self._resume_valid = False

        # ---- Stage 4: Verification (with real targeted searches)
        self.console.print(Panel("[bold magenta]Stage 4: Cross-Verifying Claims[/bold magenta]"))
        verification, verify_usage = await self._stage(
            "04_verification", lambda: self._run_verification(analysis_outputs), VerificationOutput
        )
        usage = usage + verify_usage
        self._show_usage("Verifier", verify_usage)

        # ---- Stage 5: Write Report
        self.console.print(Panel("[bold red]Stage 5: Synthesizing Report (Opus)[/bold red]"))
        if not self._budget_ok(min_tokens=10_000):
            self.console.print(
                "[yellow]Budget nearly exhausted — report quality may be reduced.[/yellow]"
            )
        try:
            report, write_usage = await self._stage(
                "05_report",
                lambda: self.writer.execute(question, verification, analysis_outputs, search_outputs),
                FinalReport,
            )
        except BudgetExhaustedError:
            self.console.print("[red]Budget exhausted during writing. Building minimal report.[/red]")
            report = self._minimal_report(question, analysis_outputs, verification)
            write_usage = TokenUsage()
        except Exception as e:
            self.console.print(f"  [red]Writer failed: {e}[/red]")
            raise
        usage = usage + write_usage
        self._show_usage("Writer", write_usage)

        self._show_budget_summary(usage)
        return report

    # ------------------------------------------------------------------ stages
    async def _run_searches(
        self, plan: ResearchPlan
    ) -> tuple[list[SearchOutput], TokenUsage]:
        search_tasks = [
            self.searcher.execute(sq, self.search_tools)
            for sq in plan.sub_questions
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        search_outputs: list[SearchOutput] = []
        search_usage = TokenUsage()
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                self.console.print(f"  [red]Searcher {i+1} failed: {result}[/red]")
                search_outputs.append(SearchOutput(
                    sub_question=plan.sub_questions[i].question,
                    results=[], total_results_found=0,
                ))
            else:
                output, u = result
                search_outputs.append(output)
                search_usage = search_usage + u
                self.console.print(
                    f"  [green]Searcher {i+1}:[/green] {output.total_results_found} results "
                    f"for '{plan.sub_questions[i].question[:50]}...'"
                )
        return search_outputs, search_usage

    async def _run_analyses(
        self, search_outputs: list[SearchOutput]
    ) -> tuple[list[AnalysisOutput], TokenUsage]:
        analysis_tasks = [self.analyzer.execute(so) for so in search_outputs]
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        analysis_outputs: list[AnalysisOutput] = []
        analysis_usage = TokenUsage()
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                self.console.print(f"  [red]Analyzer {i+1} failed: {result}[/red]")
                analysis_outputs.append(AnalysisOutput(
                    sub_question=search_outputs[i].sub_question,
                    claims=[], key_findings="Analysis failed",
                ))
            else:
                output, u = result
                analysis_outputs.append(output)
                analysis_usage = analysis_usage + u
                self.console.print(
                    f"  [green]Analyzer {i+1}:[/green] {len(output.claims)} claims extracted"
                )
        return analysis_outputs, analysis_usage

    async def _run_gap_fill(
        self,
        analysis_outputs: list[AnalysisOutput],
        search_outputs: list[SearchOutput],
    ) -> tuple[list[AnalysisOutput], TokenUsage]:
        """Second-round targeted searches for information gaps identified by analyzers."""
        all_gaps: list[str] = []
        for ao in analysis_outputs:
            all_gaps.extend(ao.gaps)

        if not all_gaps:
            return analysis_outputs, TokenUsage()

        # Cap to at most 3 gap-fill queries to control cost
        gap_queries = all_gaps[:3]
        self.console.print(
            f"  [cyan]Second round:[/cyan] filling {len(gap_queries)} information gaps"
        )

        extra_usage = TokenUsage()
        for gap in gap_queries:
            try:
                raw = await self.search_tools.search(gap, source="both")
            except Exception as e:
                logger.warning("Gap-fill search failed for '%s': %s", gap, e)
                continue
            if not raw:
                continue

            so = SearchOutput(
                sub_question=f"[gap-fill] {gap}",
                results=[],  # will be filled by analyzer from context
                total_results_found=len(raw),
            )
            # Build a synthetic SearchOutput-like context for analyzer
            # We pass raw results through a lightweight adapter
            from .models import SearchResult
            so.results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=(r.get("content") or "")[:1500],
                    full_content=(r.get("content") or "")[:5000],
                    source=r.get("source", ""),
                    relevance_score=r.get("score", 0.0),
                )
                for r in raw
            ]

            try:
                ao, u = await self.analyzer.execute(so)
                extra_usage = extra_usage + u
                analysis_outputs.append(ao)
                self.console.print(
                    f"  [green]Gap-fill:[/green] +{len(ao.claims)} claims for '{gap[:50]}'"
                )
            except Exception as e:
                logger.warning("Gap-fill analysis failed: %s", e)

        return analysis_outputs, extra_usage

    async def _run_verification(
        self, analysis_outputs: list[AnalysisOutput]
    ) -> tuple[VerificationOutput, TokenUsage]:
        try:
            verification, verify_usage = await self.verifier.execute(
                analysis_outputs, search_tools=self.search_tools
            )
            self.console.print(
                f"  Verified: {len(verification.verified_claims)} claims, "
                f"confidence: {verification.overall_confidence:.0%}"
            )
            return verification, verify_usage
        except BudgetExhaustedError as e:
            self.console.print(f"  [yellow]Budget exhausted during verification: {e}[/yellow]")
            return self._fallback_verification(analysis_outputs), TokenUsage()
        except Exception as e:
            self.console.print(f"  [yellow]Verifier failed: {e}. Using fallback.[/yellow]")
            return self._fallback_verification(analysis_outputs), TokenUsage()

    @staticmethod
    def _fallback_verification(analysis_outputs: list[AnalysisOutput]) -> VerificationOutput:
        fallback_claims = []
        for ao in analysis_outputs:
            for claim in ao.claims:
                fallback_claims.append(VerifiedClaim(
                    claim_text=claim.text,
                    source_urls=claim.source_urls,
                    verification_status="unverified",
                    notes="Auto-generated (verifier unavailable)",
                ))
        return VerificationOutput(
            verified_claims=fallback_claims,
            overall_confidence=0.5,
            recommendations="Verification was unavailable; treat claims with caution.",
        )

    @staticmethod
    def _minimal_report(
        question: str,
        analysis_outputs: list[AnalysisOutput],
        verification: VerificationOutput,
    ) -> FinalReport:
        """Build a minimal report when the budget is exhausted before writing."""
        from .models import ReportSection, Citation

        sections = []
        citations: list[Citation] = []
        seen_urls: set[str] = set()
        citation_num = 1

        for ao in analysis_outputs:
            content_lines = [f"**Key findings:** {ao.key_findings}\n"]
            for claim in ao.claims:
                content_lines.append(f"- {claim.text} (confidence: {claim.confidence})")
                for url in claim.source_urls:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(Citation(
                            number=citation_num, title=url, url=url,
                        ))
                        citation_num += 1
            sections.append(ReportSection(
                heading=ao.sub_question or "Findings",
                content="\n".join(content_lines),
                confidence="low",
            ))

        summary_parts = ["This report was auto-generated after the token budget was exhausted."]
        if verification.recommendations:
            summary_parts.append(verification.recommendations)

        return FinalReport(
            title=f"Research Report: {question[:80]}",
            executive_summary=" ".join(summary_parts),
            sections=sections,
            conclusions="See individual sections. Verification incomplete due to budget limits.",
            citations=citations,
            methodology_note="Budget-exhausted fallback report. Full pipeline did not complete.",
        )


def format_report_markdown(report: FinalReport) -> str:
    lines = [f"# {report.title}\n"]
    lines.append(f"## Executive Summary\n{report.executive_summary}\n")
    for section in report.sections:
        lines.append(f"## {section.heading}\n")
        lines.append(f"*Confidence: {section.confidence}*\n")
        lines.append(f"{section.content}\n")
    lines.append(f"## Conclusions\n{report.conclusions}\n")
    if report.citations:
        lines.append("## Citations\n")
        for c in report.citations:
            lines.append(f"[{c.number}] {c.title} — {c.url}\n")
    if report.methodology_note:
        lines.append(f"## Methodology\n{report.methodology_note}\n")
    return "\n".join(lines)
