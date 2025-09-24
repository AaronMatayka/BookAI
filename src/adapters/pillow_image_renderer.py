from __future__ import annotations
import re
from pathlib import Path
from typing import List
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

from src.ports.image_renderer import ImageComposer

class PillowImageComposer(ImageComposer):
    def _wrap(self, text: str, width: int, font) -> List[str]:
        lines, words = [], text.split()
        while words:
            line = ""
            while words and font.getlength(line + words[0]) <= width:
                line += words.pop(0) + " "
            lines.append(line.strip())
        return lines

    def compose_with_caption(self, img_path: Path, caption: str, out_path: Path) -> None:
        if Image is None:
            raise RuntimeError("`Pillow` is required: pip install pillow")
        try:
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            font_size = max(14, W // 60)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            clean = re.sub(r"\s+"," ",caption).strip()
            lines = self._wrap(clean, W-20, font)
            line_h = font.getbbox("A")[3] + 4
            cap_h = 20 + line_h * len(lines)
            from PIL import Image as PILImage
            canvas = PILImage.new("RGB",(W,H+cap_h),"white")
            canvas.paste(im,(0,0))
            draw = ImageDraw.Draw(canvas)
            y = H + 10
            for ln in lines:
                draw.text((10,y), ln, fill=(0,0,0), font=font)
                y += line_h
            canvas.save(out_path, quality=95)
        except Exception:
            if img_path.resolve() != out_path.resolve():
                out_path.write_bytes(img_path.read_bytes())
