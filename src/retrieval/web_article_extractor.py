from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WebArticle:
    url: str
    title: str = ""
    text: str = ""
    author: str | None = None
    published_at: str | None = None
    source_name: str | None = None
    canonical_url: str | None = None
    image_urls: list[str] | None = None


class WebArticleExtractor:
    def extract(self, url: str, timeout: float = 10.0) -> WebArticle:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=timeout, headers={"User-Agent": "SEMV research bot"})
        response.raise_for_status()
        html = response.text
        text = ""
        try:
            import trafilatura

            text = trafilatura.extract(html, url=url) or ""
        except Exception:
            pass
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        canonical = soup.find("link", rel="canonical")
        og_image = soup.find("meta", property="og:image")
        images = []
        if og_image and og_image.get("content"):
            images.append(og_image["content"])
        return WebArticle(
            url=url,
            title=title,
            text=text,
            canonical_url=canonical.get("href") if canonical else url,
            image_urls=images,
        )
