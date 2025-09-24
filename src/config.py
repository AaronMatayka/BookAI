from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


CONFIG_DIR = Path.home() / ".config" / "src"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = CONFIG_DIR / "config.json"
BANK_PATH   = CONFIG_DIR / "context_bank.json"


@dataclass
class Config:
    # IO
    input_pdf: Optional[Path] = None
    output_dir: Path = Path.cwd() / "bookai_images"
    workflow_file: Optional[Path] = None
    context_file: Optional[Path] = None

    # Behavior
    pages_range: str = ""
    skip_existing: bool = True
    add_text_under_image: bool = True
    summarise: bool = True
    token_cap: int = 75
    learn_descriptors: bool = True
    reset_context: bool = False

    # Services
    comfy_url: str = os.environ.get("BOOKAI_COMFY_URL", "http://127.0.0.1:8188")
    openai_key: str = os.environ.get("OPENAI_API_KEY", "")
    ollama_url: str = os.environ.get("BOOKAI_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("BOOKAI_OLLAMA_MODEL", "")

    # Optional node selection
    clip_node_id: str = ""


class ConfigManager:
    """Load / save app configuration as JSON. Keeps the rest of the code simple."""

    @staticmethod
    def load() -> Config:
        c = Config()
        if not CONFIG_PATH.exists():
            return c
        try:
            d = json.loads(CONFIG_PATH.read_text())
            def p(k: str) -> Optional[Path]:
                v = d.get(k) or ""
                return Path(v) if v else None
            c.input_pdf = p("input_pdf")
            c.output_dir = Path(d.get("output_dir", str(c.output_dir)))
            c.workflow_file = p("workflow_file")
            c.context_file = p("context_file")
            c.pages_range = d.get("pages_range", c.pages_range)
            c.skip_existing = bool(d.get("skip_existing", c.skip_existing))
            c.add_text_under_image = bool(d.get("add_text_under_image", c.add_text_under_image))
            c.summarise = bool(d.get("summarise", c.summarise))
            c.token_cap = int(d.get("token_cap", c.token_cap))
            c.learn_descriptors = bool(d.get("learn_descriptors", c.learn_descriptors))
            c.reset_context = bool(d.get("reset_context", c.reset_context))
            c.comfy_url = d.get("comfy_url", c.comfy_url)
            c.openai_key = d.get("openai_key", c.openai_key)
            c.ollama_url = d.get("ollama_url", c.ollama_url)
            c.ollama_model = d.get("ollama_model", c.ollama_model)
            c.clip_node_id = d.get("clip_node_id", c.clip_node_id)
        except Exception:
            pass
        return c

    @staticmethod
    def save(c: Config) -> None:
        data: Dict[str, Any] = {
            "input_pdf": str(c.input_pdf) if c.input_pdf else "",
            "output_dir": str(c.output_dir),
            "workflow_file": str(c.workflow_file) if c.workflow_file else "",
            "context_file": str(c.context_file) if c.context_file else "",
            "pages_range": c.pages_range,
            "skip_existing": c.skip_existing,
            "add_text_under_image": c.add_text_under_image,
            "summarise": c.summarise,
            "token_cap": c.token_cap,
            "learn_descriptors": c.learn_descriptors,
            "reset_context": c.reset_context,
            "comfy_url": c.comfy_url,
            "openai_key": c.openai_key,
            "ollama_url": c.ollama_url,
            "ollama_model": c.ollama_model,
            "clip_node_id": c.clip_node_id,
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))


# small helpers used across the app
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_pages_spec(spec: str, total: int) -> List[int]:
    if not spec:
        return list(range(1, total + 1))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = map(int, part.split("-", 1))
                if a > b: a, b = b, a
                out.extend([i for i in range(a, b + 1) if 1 <= i <= total])
            except ValueError:
                continue
        else:
            try:
                i = int(part)
                if 1 <= i <= total: out.append(i)
            except ValueError:
                pass
    return sorted(set(out))
