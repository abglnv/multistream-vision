import logging

import aiohttp

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 14; Mobile; rv:109.0) Gecko/120.0 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
]


async def _fetch_one(user_agent: str) -> tuple[str, str]:
    jar = aiohttp.CookieJar()
    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        async with session.get(
            "https://kaztoll.kz/",
            headers={"User-Agent": user_agent, "Referer": "https://kaztoll.kz/"},
            allow_redirects=True,
        ) as resp:
            await resp.read()

        cookies = jar.filter_cookies("https://kaztoll.kz/")
        cookie_str = "; ".join(f"{k}={v.value}" for k, v in cookies.items())
        logger.info(f"session ready — {len(cookie_str)} chars  UA={user_agent[:50]}")
        return cookie_str, user_agent


async def fetch_sessions(n: int) -> list[tuple[str, str]]:
    import asyncio
    tasks = [_fetch_one(USER_AGENTS[i % len(USER_AGENTS)]) for i in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sessions = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(f"session {i} failed ({r}), using empty cookie")
            sessions.append(("", USER_AGENTS[i % len(USER_AGENTS)]))
        else:
            sessions.append(r)
    return sessions
