from __future__ import annotations
from src.ports.logger import Logger

class StdoutLogger(Logger):
    def __init__(self, sink=None):
        self._sink = sink  # optional callback for web UI
    def log(self, message: str) -> None:
        if self._sink:
            self._sink(message)
        print(message)
