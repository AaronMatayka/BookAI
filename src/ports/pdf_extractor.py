from __future__ import annotations
from pathlib import Path
from typing import Protocol, List

class PDFExtractor(Protocol):
    def extract_pages(self, pdf_path: Path) -> List[str]: ...
