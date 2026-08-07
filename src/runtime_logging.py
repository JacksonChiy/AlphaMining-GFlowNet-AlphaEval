from __future__ import annotations

import io
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class TeeTextIO(io.TextIOBase):
    """Write text to the original console and a line-buffered log file."""

    def __init__(
        self,
        console: TextIO,
        log_file: TextIO,
        lock: threading.RLock,
    ) -> None:
        self.console = console
        self.log_file = log_file
        self.lock = lock

    @property
    def encoding(self) -> str:
        return getattr(self.console, "encoding", None) or "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.console.fileno()

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        with self.lock:
            self.console.write(text)
            self.log_file.write(text)
            if "\n" in text:
                self.console.flush()
                self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.log_file.flush()


def build_training_log_path(
    mode: str,
    log_dir: str | Path = "results/logs",
    log_file: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    if log_file is not None:
        return Path(log_file)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return Path(log_dir) / f"cpu_training_{mode}_{timestamp}_pid{os.getpid()}.log"


@contextmanager
def tee_console_output(path: str | Path) -> Iterator[Path]:
    """Keep terminal output while persisting stdout and stderr to one UTF-8 file."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    lock = threading.RLock()
    with target.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeTextIO(original_stdout, log_file, lock)
        sys.stderr = TeeTextIO(original_stderr, log_file, lock)
        try:
            yield target
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr
