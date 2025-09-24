from __future__ import annotations
from typing import Protocol

class Logger(Protocol):
    def log(self, message: str) -> None: ...
