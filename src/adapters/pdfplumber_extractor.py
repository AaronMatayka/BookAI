from __future__ import annotations
from pathlib import Path
from typing import List
from src.ports.pdf_extractor import PDFExtractor

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

class PdfPlumberExtractor(PDFExtractor):
    def extract_pages(self, pdf_path: Path) -> List[str]:
        if pdfplumber is None:
            raise RuntimeError("`pdfplumber` is required: pip install pdfplumber")
        texts: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                texts.append((pg.extract_text(x_tolerance=1, y_tolerance=3) or "").strip())
        return texts
