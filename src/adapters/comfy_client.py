from __future__ import annotations
import json, time
from pathlib import Path
from typing import Callable, Optional

try:
    import requests
except ImportError:
    requests = None

from src.ports.image_renderer import ImageRenderer

class ComfyClient(ImageRenderer):
    def __init__(self, base_url: str):
        if requests is None:
            raise RuntimeError("`requests` is required: pip install requests")
        self.base = base_url.rstrip("/")

    def render(self,
               workflow_file: Path,
               prompt_text: str,
               out_dir: Path,
               seed: int,
               clip_node_id: Optional[str],
               progress: Callable[[str], None]) -> Path:
        wf = json.loads(workflow_file.read_text())
        graph = wf["prompt"] if "prompt" in wf else wf

        targets = [str(clip_node_id)] if clip_node_id else None
        touched = 0
        for nid, node in graph.items():
            if "CLIPTextEncode" in node.get("class_type","") and "text" in (node.get("inputs",{}) or {}):
                if targets and str(nid) not in targets: continue
                node["inputs"]["text"] = prompt_text
                touched += 1
        if touched == 0:
            raise RuntimeError("No CLIPTextEncode node with 'text' input found.")
        progress(f"  > Injected prompt into {touched} node(s); seed={seed}")

        for node in graph.values():
            if "KSampler" in node.get("class_type",""):
                node["inputs"]["seed"] = int(seed)

        res = requests.post(f"{self.base}/prompt", json={"prompt": graph, "client_id":"BookAI"}, timeout=60)
        res.raise_for_status()
        pid = res.json()["prompt_id"]

        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(2.0)
            hist = requests.get(f"{self.base}/history/{pid}", timeout=30)
            if hist.status_code != 200:
                continue
            history = hist.json()
            if not history or pid not in history:
                continue
            outputs = history[pid].get("outputs",{})
            for node_output in outputs.values():
                if "images" in node_output:
                    img = node_output["images"][0]
                    url = (f"{self.base}/view?filename={img['filename']}"
                           f"&subfolder={img.get('subfolder','')}&type={img['type']}")
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    out = out_dir / f"temp_{int(time.time())}.png"
                    out.write_bytes(r.content)
                    return out
        raise RuntimeError("Image generation timed out.")
