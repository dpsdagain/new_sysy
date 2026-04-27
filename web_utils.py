import httpx
import logging
import re
from typing import Dict, Any, Optional
try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)

class WebFetcher:
    """
    Anthropic-Grade Web Content Extractor (F-10).
    Surgically extracts documentation and summaries from URLs.
    """
    
    @staticmethod
    def fetch_markdown(url: str, extraction_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches a URL and converts it to clean Markdown (WebFetchTool.ts:208).
        """
        try:
            # 🛡️ 1. Fetch Phase (with browser-like headers)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Antigravity/1.0"
            }
            with httpx.Client(follow_redirects=True, timeout=10.0, headers=headers) as client:
                res = client.get(url)
                res.raise_for_status()
                
            # 🧬 2. Transform Phase (trafilatura: strips sidebars/ads)
            if trafilatura:
                content = trafilatura.extract(res.text, include_links=True, include_formatting=True)
            else:
                # Fallback to simple regex-based stripping if library is missing
                content = re.sub(r'<script.*?</script>', '', res.text, flags=re.DOTALL)
                content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<.*?>', '', content)

            if not content:
                content = "Could not extract meaningful content from the page."

            # ✂️ 3. Token Density Phase (WebFetchTool.ts:271)
            # If the user provides a prompt, we should ideally use a fast model.
            # For this standalone POC, we'll return the full distilled content.
            return {
                "url": url,
                "content": content[:15000], # Parity cap (MAX_MARKDOWN_LENGTH)
                "status": res.status_code,
                "length": len(content)
            }
        except Exception as e:
            logger.error(f"WebFetch failed for {url}: {e}")
            return {"error": str(e), "url": url}
