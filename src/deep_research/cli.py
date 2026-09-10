"""CLI entry point for deep-research."""
import asyncio
import argparse
import logging
import sys
import traceback

from rich.console import Console
from rich.panel import Panel

from .config import Settings
from .pipeline import ResearchPipeline, format_report_markdown


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


async def run():
    parser = argparse.ArgumentParser(description="Deep Research: Multi-Agent AI Research System")
    parser.add_argument("question", nargs="?", default=None, help="Research question")
    parser.add_argument("--budget", type=int, default=500_000, help="Max total tokens (default: 500000)")
    parser.add_argument("--output", type=str, default=None, help="Save report to markdown file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Reuse an existing run ID for checkpointing (enables resume-friendly runs)",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)
    console = Console()

    if not args.question:
        console.print("[bold cyan]Deep Research — Multi-Agent AI Research System[/bold cyan]\n")
        args.question = input("Enter your research question: ").strip()
        if not args.question:
            console.print("[red]No question provided. Exiting.[/red]")
            return

    settings = Settings()
    settings.max_total_tokens = args.budget

    # Fail fast on missing credentials
    if not settings.anthropic_api_key:
        console.print(
            "[red]Error: DR_ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key.[/red]"
        )
        sys.exit(1)
    if not settings.tavily_api_key and not settings.firecrawl_api_key:
        console.print(
            "[red]Error: set at least one of DR_TAVILY_API_KEY / DR_FIRECRAWL_API_KEY.[/red]"
        )
        sys.exit(1)

    console.print(f"\n[bold]Question:[/bold] {args.question}")
    console.print(f"[bold]Token Budget:[/bold] {args.budget:,}\n")

    pipeline = ResearchPipeline(settings, run_id=args.run_id)
    report = await pipeline.run(args.question)

    console.print("\n" + "=" * 60)
    console.print(Panel(f"[bold]{report.title}[/bold]", style="bold green"))
    md = format_report_markdown(report)
    console.print(md)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        console.print(f"\n[green]Report saved to {args.output}[/green]")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
