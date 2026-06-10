import httpx
import asyncio
import logging

from .config import Settings

logger = logging.getLogger("deep_research.search")


class SearchTools:
    """Async HTTP clients for Tavily and Firecrawl search APIs."""

    def __init__(self, settings: Settings):
        self.tavily_key = settings.tavily_api_key
        self.firecrawl_key = settings.firecrawl_api_key
        self.max_results = settings.search_results_per_query
        self.max_scrape_chars = settings.firecrawl_scrape_max_chars

    async def tavily_search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_key,
                    "query": query,
                    "max_results": self.max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "source": "tavily",
                }
                for r in data.get("results", [])
            ]

    async def firecrawl_search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {self.firecrawl_key}"},
                json={
                    "query": query,
                    "limit": self.max_results,
                    "scrapeOptions": {"formats": ["markdown"]},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("metadata", {}).get("title", r.get("title", "")),
                    "url": r.get("url", r.get("metadata", {}).get("sourceURL", "")),
                    "content": (r.get("markdown", "") or "")[: self.max_scrape_chars],
                    "score": 0.5,
                    "source": "firecrawl",
                }
                for r in data.get("data", [])
            ]

    async def search(self, query: str, source: str = "both") -> list[dict]:
        """Unified search across Tavily and/or Firecrawl."""
        tasks = []
        if source in ("tavily", "both"):
            tasks.append(self.tavily_search(query))
        if source in ("firecrawl", "both"):
            tasks.append(self.firecrawl_search(query))

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        merged = []
        for result in results_lists:
            if isinstance(result, Exception):
                logger.warning("Search source failed: %s", result)
                continue
            merged.extend(result)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        deduplicated = []
        for r in merged:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                deduplicated.append(r)

        return sorted(deduplicated, key=lambda x: x.get("score", 0), reverse=True)
