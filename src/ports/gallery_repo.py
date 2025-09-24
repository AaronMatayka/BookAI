from __future__ import annotations
from pathlib import Path
from typing import Protocol, List

class GalleryRepository(Protocol):
    def list_images(self, folder: Path) -> List[Path]: ...
