import asyncio
import argparse
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from deep_research.config import Settings
from deep_research.pipeline import ResearchPipeline, format_report_markdown
from rich.console import Console
from rich.panel import Panel


async def run():
    parser = argparse.ArgumentParser(description="Deep Research: Multi-Agent AI Research System")
    parser.add_argument("question", nargs="?", default=None, help="Research question")
    parser.add_argument("--budget", type=int, default=500_000, help="Max total tokens (default: 500000)")
    parser.add_argument("--output", type=str, default=None, help="Save report to markdown file")
    args = parser.parse_args()

    console = Console()

    if not args.question:
        console.print("[bold cyan]Deep Research — Multi-Agent AI Research System[/bold cyan]\n")
        args.question = input("Enter your research question: ").strip()
        if not args.question:
            console.print("[red]No question provided. Exiting.[/red]")
            return

    settings = Settings()
    settings.max_total_tokens = args.budget

    console.print(f"\n[bold]Question:[/bold] {args.question}")
    console.print(f"[bold]Token Budget:[/bold] {args.budget:,}\n")

    pipeline = ResearchPipeline(settings)
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
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
    finally:
        # Keep terminal open on Windows (only in interactive mode)
        if sys.platform == "win32" and sys.stdin.isatty():
            input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
