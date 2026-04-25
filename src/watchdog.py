"""
Phase 5 — Python watchdog.

Launches the orchestrator as a child process and restarts it when:
  • it dies
  • RAM exceeds MEMORY_LIMIT_MB
  • CPU stays at 0% for CPU_ZERO_WINDOW seconds (deadlock)
"""

import logging
import subprocess
import sys
import time

import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [watchdog] %(message)s")
logger = logging.getLogger(__name__)

MEMORY_LIMIT_MB  = 2048
CPU_ZERO_THRESHOLD = 0.5   # below counts as "zero"
CPU_ZERO_WINDOW    = 30    # seconds of 0% CPU before we deadlock
POLL_INTERVAL      = 5     
RESTART_CMD        = [sys.executable, "-m", "src.main"]


def launch() -> psutil.Process:
    proc = subprocess.Popen(RESTART_CMD)
    logger.info(f"launched orchestrator PID={proc.pid}")
    return psutil.Process(proc.pid)

def main() -> None:
    proc = launch()
    zero_cpu_since: float | None = None

    while True:
        time.sleep(POLL_INTERVAL)

        try:
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                logger.warning("orchestrator is dead — restarting")
                proc = launch()
                zero_cpu_since = None 
                continue 

            mem_mb = proc.memory_info().rss / 1_048_576 
            cpu = proc.cpu_percent(interval = 1)
            logger.info(f"PID={proc.pid}  RAM={mem_mb:.0f} MB  CPU={cpu:.1f}%")

            if mem_mb > MEMORY_LIMIT_MB:
                logger.error(f"memory limit ({MEMORY_LIMIT_MB} MB) exceeded — killing")
                proc.kill()
                proc = launch()
                zero_cpu_since = None 
                continue 

            if cpu < CPU_ZERO_THRESHOLD:
                zero_cpu_since = zero_cpu_since or time.time()
                if time.time() - zero_cpu_since > CPU_ZERO_WINDOW:
                    logger.error(f"deadlock: 0% CPU for {CPU_ZERO_WINDOW} s — killing")
                    proc.kill()
                    proc = launch()
                    zero_cpu_since = None 
            else:
                zero_cpu_since = None 

        except psutil.NoSuchProcess:
            logger.warning("process vanished — restarting")
            proc = launch()
            zero_cpu_since = None


if __name__ == "__main__":
    main()
