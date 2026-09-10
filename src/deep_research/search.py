import asyncio
import logging
import random

import httpx

from .config import Settings

logger = logging.getLogger("deep_research.search")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SearchTools:
    """Async HTTP clients for Tavily and Firecrawl search APIs, with retries and concurrency limits."""

    def __init__(self, settings: Settings, semaphore: asyncio.Semaphore | None = None):
        self.tavily_key = settings.tavily_api_key
        self.firecrawl_key = settings.firecrawl_api_key
        self.max_results = settings.search_results_per_query
        self.max_scrape_chars = settings.firecrawl_scrape_max_chars
        self.max_retries = settings.search_max_retries
        self.retry_base_delay = settings.search_retry_base_delay
        self.semaphore = semaphore or asyncio.Semaphore(settings.search_concurrency)

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=kwargs.pop("timeout", 30)) as client:
                    resp = await client.request(method, url, **kwargs)
                    if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                        last_exc = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                    else:
                        resp.raise_for_status()
                        return resp
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= self.max_retries:
                    raise
                last_exc = e

            delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            logger.warning(
                "Search request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, self.max_retries + 1, last_exc, delay,
            )
            await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover

    async def tavily_search(self, query: str) -> list[dict]:
        async with self.semaphore:
            resp = await self._request_with_retry(
                "POST",
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_key,
                    "query": query,
                    "max_results": self.max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                },
                timeout=30,
            )
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
        async with self.semaphore:
            resp = await self._request_with_retry(
                "POST",
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {self.firecrawl_key}"},
                json={
                    "query": query,
                    "limit": self.max_results,
                    "scrapeOptions": {"formats": ["markdown"]},
                },
                timeout=60,
            )
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

    async def scrape_url(self, url: str) -> str:
        """Deep-scrape a single URL via Firecrawl, returning markdown content."""
        if not self.firecrawl_key:
            return ""
        try:
            async with self.semaphore:
                resp = await self._request_with_retry(
                    "POST",
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={"Authorization": f"Bearer {self.firecrawl_key}"},
                    json={"url": url, "formats": ["markdown"]},
                    timeout=60,
                )
                data = resp.json()
                markdown = (data.get("data", {}) or {}).get("markdown", "") or ""
                return markdown[: self.max_scrape_chars]
        except Exception as e:
            logger.warning("Deep scrape failed for %s: %s", url, e)
            return ""

    async def deep_scrape_top(self, results: list[dict], top_n: int) -> list[dict]:
        """Scrape full page content for the top-N results (by score). Fill in 'content'."""
        if top_n <= 0 or not results:
            return results

        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        to_scrape = sorted_results[:top_n]
        rest = sorted_results[top_n:]

        scrape_tasks = [self.scrape_url(r["url"]) for r in to_scrape]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        enriched = []
        for r, content in zip(to_scrape, scraped_contents):
            item = dict(r)
            if isinstance(content, Exception) or not content:
                # Keep the original snippet if scrape failed
                pass
            else:
                item["content"] = content
                item["scraped"] = True
            enriched.append(item)

        for r in rest:
            item = dict(r)
            item["scraped"] = False
            enriched.append(item)

        return enriched

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
