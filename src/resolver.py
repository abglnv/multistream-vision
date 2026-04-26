import asyncio
import logging

logger = logging.getLogger(__name__)


def _is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _resolve(url: str) -> str:
    import yt_dlp

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info["url"]
        logger.info(f"resolved {url[:50]}... → {stream_url[:80]}...")
        return stream_url


async def resolve_urls(urls: list[str]) -> list[str]:
    """Resolve any YouTube URLs in the list; pass others through unchanged."""
    resolved = []
    for url in urls:
        if _is_youtube(url):
            try:
                stream_url = await asyncio.to_thread(_resolve, url)
                resolved.append(stream_url)
            except Exception as exc:
                logger.error(f"failed to resolve {url}: {exc}")
                resolved.append(url)  # fall back to original
        else:
            resolved.append(url)
    return resolved
