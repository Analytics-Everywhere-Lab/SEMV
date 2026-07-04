from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup


class WebImageCandidateExtractor:
    def extract_image_urls(self, page_url: str, html: str, limit: int = 20) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        for image in soup.find_all("img"):
            src = image.get("src") or image.get("data-src")
            if src:
                urls.append(urljoin(page_url, src))
            if len(urls) >= limit:
                break
        return urls
