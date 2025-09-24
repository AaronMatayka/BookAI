from __future__ import annotations
from pathlib import Path
from typing import Protocol, Optional, Callable

class ImageRenderer(Protocol):
    def render(self,
               workflow_file: Path,
               prompt_text: str,
               out_dir: Path,
               seed: int,
               clip_node_id: Optional[str],
               progress: Callable[[str], None]) -> Path: ...

class ImageComposer(Protocol):
    def compose_with_caption(self, img_path: Path, caption: str, out_path: Path) -> None: ...
