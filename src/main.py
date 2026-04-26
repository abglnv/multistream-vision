import argparse
import asyncio
import logging
import os
import signal

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from src.consumer import InferenceEngine, inference_consumer, start_web
from src.producer import QUEUE_MAXSIZE, stream_producer
from src.resolver import resolve_urls
from src.streams import STREAMS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run(stream_urls: list[str]) -> None:
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("resolving stream URLs...")
    stream_urls = await resolve_urls(stream_urls)

    queues = [asyncio.Queue(maxsize=QUEUE_MAXSIZE) for _ in stream_urls]
    engine = InferenceEngine()

    producers = [
        asyncio.create_task(
            stream_producer(i, url, queues[i], stop_event),
            name=f"producer-{i}",
        )
        for i, url in enumerate(stream_urls)
    ]
    consumer = asyncio.create_task(
        inference_consumer(queues, engine, stop_event),
        name="consumer",
    )

    await start_web()
    logger.info(f"Orchestrator running — {len(stream_urls)} streams")
    await asyncio.gather(*producers, consumer, return_exceptions=True)
    logger.info("Orchestrator stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=len(STREAMS),
                        help="Number of streams to use (default: all)")
    args = parser.parse_args()
    asyncio.run(run(STREAMS[: args.streams]))


if __name__ == "__main__":
    main()
