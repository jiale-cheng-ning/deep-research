import asyncio
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .llm import LLMClient, TokenBudget, TokenUsage
from .models import FinalReport, ResearchPlan, SearchOutput, AnalysisOutput, VerificationOutput
from .search import SearchTools
from .agents.orchestrator import OrchestratorAgent
from .agents.searcher import SearcherAgent
from .agents.analyzer import AnalyzerAgent
from .agents.verifier import VerifierAgent
from .agents.writer import WriterAgent

logger = logging.getLogger("deep_research.pipeline")


class ResearchPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.budget = TokenBudget(max_total=settings.max_total_tokens)
        self.llm = LLMClient(settings, self.budget)
        self.search_tools = SearchTools(settings)
        self.console = Console()

        self.orchestrator = OrchestratorAgent(self.llm, settings)
        self.searcher = SearcherAgent(self.llm, settings)
        self.analyzer = AnalyzerAgent(self.llm, settings)
        self.verifier = VerifierAgent(self.llm, settings)
        self.writer = WriterAgent(self.llm, settings)

    async def run(self, question: str) -> FinalReport:
        # Stage 1: Plan
        self.console.print(Panel("[bold cyan]Stage 1: Planning Research Strategy[/bold cyan]"))
        plan, plan_usage = await self.orchestrator.execute(question)
        self._show_plan(plan)
        self._show_usage("Orchestrator", plan_usage)

        # Stage 2: Parallel Search
        self.console.print(Panel("[bold yellow]Stage 2: Searching (parallel)[/bold yellow]"))
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
                    sub_question=plan.sub_questions[i].question, results=[], total_results_found=0
                ))
            else:
                output, usage = result
                search_outputs.append(output)
                search_usage = search_usage + usage
                self.console.print(
                    f"  [green]Searcher {i+1}:[/green] {output.total_results_found} results "
                    f"for '{plan.sub_questions[i].question[:50]}...'"
                )
        self._show_usage("Searchers", search_usage)

        # Stage 3: Parallel Analysis
        self.console.print(Panel("[bold green]Stage 3: Analyzing Claims (parallel)[/bold green]"))
        analysis_tasks = [self.analyzer.execute(so) for so in search_outputs]
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        analysis_outputs: list[AnalysisOutput] = []
        analysis_usage = TokenUsage()
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                self.console.print(f"  [red]Analyzer {i+1} failed: {result}[/red]")
                analysis_outputs.append(AnalysisOutput(
                    sub_question=search_outputs[i].sub_question, claims=[], key_findings="Analysis failed"
                ))
            else:
                output, usage = result
                analysis_outputs.append(output)
                analysis_usage = analysis_usage + usage
                self.console.print(
                    f"  [green]Analyzer {i+1}:[/green] {len(output.claims)} claims extracted"
                )
        self._show_usage("Analyzers", analysis_usage)

        # Stage 4: Verification
        self.console.print(Panel("[bold magenta]Stage 4: Cross-Verifying Claims[/bold magenta]"))
        try:
            verification, verify_usage = await self.verifier.execute(analysis_outputs)
            self.console.print(
                f"  Verified: {len(verification.verified_claims)} claims, "
                f"confidence: {verification.overall_confidence:.0%}"
            )
        except Exception as e:
            self.console.print(f"  [yellow]Verifier failed: {e}. Using fallback.[/yellow]")
            # Build a basic verification from analysis outputs
            from .models import VerifiedClaim
            fallback_claims = []
            for ao in analysis_outputs:
                for claim in ao.claims:
                    fallback_claims.append(VerifiedClaim(
                        claim_text=claim.text,
                        source_urls=claim.source_urls,
                        verification_status="unverified",
                        notes="Auto-generated (verifier unavailable)",
                    ))
            verification = VerificationOutput(
                verified_claims=fallback_claims,
                overall_confidence=0.5,
                recommendations="Verification was unavailable; treat claims with caution.",
            )
            verify_usage = TokenUsage()
        self._show_usage("Verifier", verify_usage)

        # Stage 5: Write Report
        self.console.print(Panel("[bold red]Stage 5: Synthesizing Report (Opus)[/bold red]"))
        try:
            report, write_usage = await self.writer.execute(
                question, verification, analysis_outputs, search_outputs
            )
        except Exception as e:
            self.console.print(f"  [red]Writer failed: {e}[/red]")
            raise
        self._show_usage("Writer", write_usage)

        # Summary
        total = plan_usage + search_usage + analysis_usage + verify_usage + write_usage
        self._show_budget_summary(total)

        return report

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
        pct = self.budget.usage.total / self.budget.max_total * 100
        table.add_row("Budget Used", f"{pct:.1f}%")
        self.console.print(table)


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
