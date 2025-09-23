#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# --- Dependency Imports ---
# These are wrapped in try/except blocks to provide helpful error messages
# if a library is missing.

# PDF Processing
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Imaging
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

# HTTP Requests
try:
    import requests
except ImportError:
    requests = None

# OpenAI (optional for summarization)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Local Context Engine
from context_bank import ContextBank, DescriptorExtractor, enrich_prompt_with_context

# Ollama integration: these may fail to import if the module is missing.
try:
    from ollama_connector import (
        ollama_extract_descriptors,
        list_models as list_ollama_models,
        ollama_generate_prompt,
    )
except Exception:
    # Provide fallbacks if the connector is unavailable
    def ollama_extract_descriptors(model: str, text: str, url: str = "http://localhost:11434"):
        return []


    def list_ollama_models(url: str = "http://localhost:11434"):
        return []


    def ollama_generate_prompt(model: str, context: str, text: str, url: str = "http://localhost:11434") -> str:
        return ""

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Configuration ---
CONFIG_DIR = Path.home() / ".config" / "bookai"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
BANK_PATH = CONFIG_DIR / "context_bank.json"


@dataclass
class Settings:
    """Holds all user-configurable settings for the application."""
    input_pdf: Optional[Path] = None
    output_dir: Path = Path.cwd() / "bookai_images"
    openai_key: str = os.environ.get("OPENAI_API_KEY", "")
    summarise: bool = True
    pages_range: str = ""
    skip_existing: bool = True
    token_cap: int = 75
    add_text_under_image: bool = True
    comfy_url: str = os.environ.get("BOOKAI_COMFY_URL", "http://127.0.0.1:8188")
    workflow_file: Optional[Path] = None
    context_file: Optional[Path] = None
    learn_descriptors: bool = True
    reset_context: bool = False

    # Ollama settings for descriptor extraction and prompt generation
    ollama_url: str = os.environ.get("BOOKAI_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("BOOKAI_OLLAMA_MODEL", "")

    # Node selection for prompt injection
    # Specifies the ID of the CLIPTextEncode node into which the prompt should be injected.
    # If empty, all CLIPTextEncode nodes will be injected (existing behaviour).
    clip_node_id: str = ""


# -------------------- Utilities --------------------

def ensure_dir(p: Path) -> None:
    """Ensures a directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def extract_pages(pdf_path: Path) -> List[str]:
    """Extracts text from all pages of a PDF."""
    if pdfplumber is None:
        raise RuntimeError("`pdfplumber` is required to read PDFs. Please run: pip install pdfplumber")

    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            texts.append((pg.extract_text(x_tolerance=1, y_tolerance=3) or "").strip())
    return texts


def parse_pages_spec(spec: str, total: int) -> List[int]:
    """Parses a page range string (e.g., '1-5, 8, 10-12') into a list of page numbers."""
    if not spec:
        return list(range(1, total + 1))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
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
    return sorted(list(set(out)))


def export_context_snapshot(bank: ContextBank, dest: Path) -> None:
    """Writes a human-readable snapshot of the context bank to a text file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BookAI Context Snapshot",
        f"# Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "#" + "-" * 50
    ]
    entities = bank.data.get("entities", {})
    if not entities:
        lines.append("\n# Context bank is empty.")
    else:
        for name, meta in sorted(entities.items()):
            desc = (meta.get("description") or "").strip()
            aliases = [a for a in (meta.get("aliases") or []) if a]
            line = f"## {name}\n"
            if desc:
                line += f"- Description: {desc}\n"
            if aliases:
                line += f"- Aliases: {', '.join(aliases)}\n"
            lines.append(line)
    dest.write_text("\n".join(lines), encoding="utf-8")


def wrap_text(text: str, width: int, font: ImageFont.FreeTypeFont) -> List[str]:
    """Wraps text to fit a given pixel width."""
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and font.getlength(line + words[0]) <= width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines


def compose_with_caption(img_path: Path, caption: str, out_path: Path) -> None:
    """Adds a text caption below an image."""
    if Image is None:
        raise RuntimeError("`Pillow` is required for image operations. Please run: pip install Pillow")

    try:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        font_size = max(14, W // 60)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Clean and wrap caption text
        clean_caption = re.sub(r'\s+', ' ', caption).strip()
        wrapped_lines = wrap_text(clean_caption, W - 20, font)

        # Calculate caption height
        line_h = font.getbbox("A")[3] + 4
        cap_h = 20 + line_h * len(wrapped_lines)

        canvas = Image.new("RGB", (W, H + cap_h), "white")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)

        y = H + 10
        for ln in wrapped_lines:
            draw.text((10, y), ln, fill=(0, 0, 0), font=font)
            y += line_h

        canvas.save(out_path, quality=95)

    except Exception as e:
        print(f"Failed to compose caption: {e}. Copying original image instead.")
        if img_path.resolve() != out_path.resolve():
            out_path.write_bytes(img_path.read_bytes())


# -------------------- OpenAI Summarizer (Optional) --------------------

def openai_summarise(key: str, txt: str, token_cap: int) -> str:
    """Summarizes text into a visual prompt using OpenAI's API."""
    if not key or OpenAI is None:
        return ""
    client = OpenAI(api_key=key)
    system_prompt = (
        "You are an expert at creating visual prompts for an AI image generator based on literary text. "
        "Your task is to summarize the provided book page into a single, concise paragraph. "
        "Focus on concrete visual details: the setting, character appearances, key actions, and mood. "
        "Do not use descriptive adjectives like 'beautiful' or 'interesting'. Instead, describe *why* something is beautiful (e.g., 'golden light filtering through leaves'). "
        "Retain all named characters. Mention the style if relevant (e.g., 'oil painting', 'comic book art')."
        f"The summary should be around {token_cap} words."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TEXT:\n{txt}"}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"OpenAI summarization failed: {e}")
        return ""


# -------------------- ComfyUI Integration --------------------

def comfy_inject_and_generate(
        comfy_url: str,
        workflow_file: Path,
        prompt_text: str,
        out_dir: Path,
        page_seed: int,
        log_fn,
        clip_node_id: Optional[str] = None,
) -> Path:
    """Injects a prompt into a ComfyUI workflow, runs it, and downloads the result.

    If ``clip_node_id`` is provided, the prompt will only be injected into the
    specified CLIPTextEncode node.  Otherwise, it will inject into all
    CLIPTextEncode nodes as before.
    """
    if requests is None:
        raise RuntimeError("`requests` is required for the ComfyUI backend. Please run: pip install requests")

    # 1. Load and parse the workflow, handling both raw and nested graph formats
    workflow_data = json.loads(workflow_file.read_text())
    if "prompt" in workflow_data:
        graph = workflow_data["prompt"]
    else:
        graph = workflow_data

    # 2. Inject prompt into all relevant text nodes
    touched_nodes = 0
    # Determine which node IDs to inject
    target_ids: Optional[List[str]] = None
    if clip_node_id:
        target_ids = [clip_node_id]
    for nid, node in graph.items():
        class_type = node.get("class_type", "")
        if "CLIPTextEncode" in class_type:
            # Skip if node does not have a text input
            if "text" not in node.get("inputs", {}):
                continue
            # If a specific node ID is supplied, only inject into that
            if target_ids is not None:
                if str(nid) not in target_ids:
                    continue
            node["inputs"]["text"] = prompt_text
            touched_nodes += 1
    if touched_nodes == 0:
        raise RuntimeError("Could not find a 'CLIPTextEncode' node in your workflow to inject the prompt into.")

    # 3. Set a deterministic seed for the page in all KSampler nodes
    for node in graph.values():
        if "KSampler" in node.get("class_type", ""):
            node["inputs"]["seed"] = int(page_seed)

    log_fn(f"  > Injected prompt into {touched_nodes} node(s); seed={page_seed}")

    # 4. Queue the prompt
    payload = {"prompt": graph, "client_id": "BookAI"}
    try:
        res = requests.post(f"{comfy_url.rstrip('/')}/prompt", json=payload, timeout=60)
        res.raise_for_status()
        prompt_id = res.json()["prompt_id"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not connect to ComfyUI at {comfy_url}. Is it running? Error: {e}")

    # 5. Poll history until an image is ready
    deadline = time.time() + 300  # 5-minute timeout
    while time.time() < deadline:
        time.sleep(2.0)
        hist_res = requests.get(f"{comfy_url.rstrip('/')}/history/{prompt_id}", timeout=30)
        if hist_res.status_code != 200:
            continue

        history = hist_res.json()
        if not history or prompt_id not in history:
            continue

        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                image_data = node_output['images'][0]
                image_url = f"{comfy_url.rstrip('/')}/view?filename={image_data['filename']}&subfolder={image_data.get('subfolder', '')}&type={image_data['type']}"

                img_response = requests.get(image_url, timeout=120)
                img_response.raise_for_status()

                dest = out_dir / f"temp_{int(time.time())}.png"
                dest.write_bytes(img_response.content)
                return dest

    raise RuntimeError("Image generation timed out. Check your ComfyUI console for errors.")


# -------------------- GUI Application --------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BookAI - Your Personal Book Illustrator")
        self.geometry("980x720")
        self.s = load_config()
        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Input/Output Configuration
        io_frame = ttk.LabelFrame(main_frame, text="File Configuration", padding="10")
        io_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Input PDF:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.var_pdf = tk.StringVar(value=str(self.s.input_pdf or ""))
        ttk.Entry(io_frame, textvariable=self.var_pdf).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(io_frame, text="Browse…", command=self._pick_pdf).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(io_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.var_out = tk.StringVar(value=str(self.s.output_dir))
        ttk.Entry(io_frame, textvariable=self.var_out).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(io_frame, text="Choose…", command=self._pick_out).grid(row=1, column=2, padx=5, pady=5)

        # Backend Configuration
        backend_frame = ttk.LabelFrame(main_frame, text="AI Backend (ComfyUI)", padding="10")
        backend_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        backend_frame.columnconfigure(1, weight=1)

        ttk.Label(backend_frame, text="ComfyUI URL:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.var_comfy = tk.StringVar(value=self.s.comfy_url)
        ttk.Entry(backend_frame, textvariable=self.var_comfy, width=40).grid(row=0, column=1, sticky="w", padx=5,
                                                                             pady=5)

        ttk.Label(backend_frame, text="Workflow JSON:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.var_wf = tk.StringVar(value=str(self.s.workflow_file or ""))
        ttk.Entry(backend_frame, textvariable=self.var_wf).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(backend_frame, text="Browse…", command=self._pick_wf).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(backend_frame, text="OpenAI API Key (for summarization):").grid(row=2, column=0, sticky="e", padx=5,
                                                                                  pady=5)
        self.var_key = tk.StringVar(value=self.s.openai_key)
        ttk.Entry(backend_frame, textvariable=self.var_key, show="*").grid(row=2, column=1, sticky="we", padx=5, pady=5)

        # Ollama configuration
        ttk.Label(backend_frame, text="Ollama URL:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.var_ollama_url = tk.StringVar(value=self.s.ollama_url)
        ttk.Entry(backend_frame, textvariable=self.var_ollama_url, width=40).grid(row=3, column=1, sticky="w", padx=5,
                                                                                  pady=5)

        ttk.Label(backend_frame, text="Ollama Model:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.var_ollama_model = tk.StringVar(value=self.s.ollama_model)
        self.combo_model = ttk.Combobox(backend_frame, textvariable=self.var_ollama_model, width=30, state="readonly")
        self.combo_model.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(backend_frame, text="Refresh Models", command=self._refresh_models).grid(row=4, column=2, padx=5,
                                                                                            pady=5)

        # CLIP Node Selection for prompt injection
        ttk.Label(backend_frame, text="Inject into Node:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.var_clip_node = tk.StringVar(value=self.s.clip_node_id)
        self.combo_clip_node = ttk.Combobox(backend_frame, textvariable=self.var_clip_node, width=20, state="readonly")
        self.combo_clip_node.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(backend_frame, text="Refresh Nodes", command=self._refresh_nodes).grid(row=5, column=2, padx=5,
                                                                                          pady=5)

        # Context and Generation Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Generation Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Pages (e.g., 1-5, 8):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.var_pages = tk.StringVar(value=self.s.pages_range)
        ttk.Entry(settings_frame, textvariable=self.var_pages, width=30).grid(row=0, column=1, sticky="w", padx=5,
                                                                              pady=5)

        ttk.Label(settings_frame, text="Context Snapshot File:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.var_ctx = tk.StringVar(value=str(self.s.context_file or ""))
        ttk.Entry(settings_frame, textvariable=self.var_ctx).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse…", command=self._pick_ctx).grid(row=1, column=2, padx=5, pady=5)

        # Checkboxes
        check_frame = ttk.Frame(settings_frame)
        check_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)

        self.var_learn = tk.BooleanVar(value=self.s.learn_descriptors)
        ttk.Checkbutton(check_frame, text="Auto-learn character details", variable=self.var_learn).pack(side="left",
                                                                                                        padx=5)

        self.var_reset = tk.BooleanVar(value=self.s.reset_context)
        ttk.Checkbutton(check_frame, text="Reset context before run", variable=self.var_reset).pack(side="left", padx=5)

        self.var_sum = tk.BooleanVar(value=self.s.summarise)
        ttk.Checkbutton(check_frame, text="Summarise pages (needs OpenAI key)", variable=self.var_sum).pack(side="left",
                                                                                                            padx=5)

        self.var_skip = tk.BooleanVar(value=self.s.skip_existing)
        ttk.Checkbutton(check_frame, text="Skip existing images", variable=self.var_skip).pack(side="left", padx=5)

        self.var_under = tk.BooleanVar(value=self.s.add_text_under_image)
        ttk.Checkbutton(check_frame, text="Add page text under image", variable=self.var_under).pack(side="left",
                                                                                                     padx=5)

        # Log and Generate Button
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=12, wrap="word", relief="sunken", borderwidth=1)
        self.log.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log['yscrollcommand'] = scrollbar.set

        self.generate_button = ttk.Button(main_frame, text="Generate Images", command=self._generate,
                                          style="Accent.TButton")
        self.generate_button.grid(row=4, column=2, sticky="e", pady=10)

        # Configure styles and weights
        self.style = ttk.Style(self)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Populate the Ollama model list on startup
        try:
            self._refresh_models()
        except Exception:
            pass
        # Also populate CLIP node list based on current workflow
        try:
            self._refresh_nodes()
        except Exception:
            pass

    def _pick_pdf(self):
        self.var_pdf.set(filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")]) or self.var_pdf.get())

    def _pick_out(self):
        self.var_out.set(filedialog.askdirectory() or self.var_out.get())

    def _pick_wf(self):
        """Prompt the user to select a workflow JSON file and refresh the CLIP node list."""
        selected = filedialog.askopenfilename(filetypes=[("JSON Workflow", "*.json")])
        if selected:
            self.var_wf.set(selected)
            # Refresh node choices when a new workflow is selected
            try:
                self._refresh_nodes()
            except Exception:
                pass
        else:
            # Nothing selected, retain previous value
            pass

    def _pick_ctx(self):
        self.var_ctx.set(filedialog.asksaveasfilename(defaultextension=".md", filetypes=[
            ("Markdown/Text", "*.md *.txt")]) or self.var_ctx.get())

    def log_print(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _refresh_models(self) -> None:
        """
        Fetch available Ollama models from the configured URL and update the model dropdown.
        If no URL is provided or the request fails, the dropdown is cleared.
        """
        url = self.var_ollama_url.get().strip()
        if not url:
            self.log_print("  > No Ollama URL specified; cannot fetch models.")
            self.combo_model['values'] = []
            return
        self.log_print(f"  > Connecting to Ollama at {url} to fetch model list…")
        try:
            models = list_ollama_models(url)
        except Exception as e:
            models = []
            self.log_print(f"  > (!) Failed to fetch models: {e}")
        if models:
            self.log_print(f"  > Available models: {', '.join(models)}")
            self.combo_model['values'] = models
            current = self.var_ollama_model.get().strip()
            if not current or current not in models:
                self.var_ollama_model.set(models[0])
        else:
            self.log_print("  > No models found on the Ollama server.")
            self.combo_model['values'] = []

    def _refresh_nodes(self) -> None:
        """
        Parse the selected workflow file and populate the injection node dropdown
        with the IDs of all CLIPTextEncode nodes. If no workflow is selected
        or no nodes are found, the dropdown is cleared.
        """
        wf_path = self.var_wf.get().strip()
        if not wf_path or not os.path.exists(wf_path):
            self.log_print("  > Please select a valid workflow JSON before refreshing nodes.")
            self.combo_clip_node['values'] = []
            return
        try:
            with open(wf_path, 'r', encoding='utf-8') as f:
                wf_data = json.load(f)
            graph = wf_data.get("prompt") if isinstance(wf_data.get("prompt"), dict) else wf_data
            node_ids = []
            for nid, node in graph.items():
                if not isinstance(node, dict):
                    continue
                class_type = node.get("class_type", "") or ""
                inputs = node.get("inputs", {}) or {}
                if "CLIPTextEncode" in class_type and "text" in inputs:
                    node_ids.append(str(nid))
            if node_ids:
                self.log_print(f"  > Found CLIP nodes: {', '.join(node_ids)}")
                self.combo_clip_node['values'] = node_ids
                current = self.var_clip_node.get().strip()
                if current and current in node_ids:
                    # keep current
                    pass
                else:
                    # default to first found node
                    self.var_clip_node.set(node_ids[0])
            else:
                self.log_print("  > No CLIPTextEncode nodes with text input found in workflow.")
                self.combo_clip_node['values'] = []
        except Exception as e:
            self.log_print(f"  > (!) Failed to parse workflow for nodes: {e}")
            self.combo_clip_node['values'] = []

    def _generate(self):
        s = Settings(
            input_pdf=Path(self.var_pdf.get()) if self.var_pdf.get().strip() else None,
            output_dir=Path(self.var_out.get()),
            openai_key=self.var_key.get().strip(),
            summarise=self.var_sum.get(),
            pages_range=self.var_pages.get(),
            skip_existing=self.var_skip.get(),
            token_cap=75,  # Hardcoded for now, can be UI element
            add_text_under_image=self.var_under.get(),
            comfy_url=self.var_comfy.get().strip(),
            workflow_file=Path(self.var_wf.get()) if self.var_wf.get().strip() else None,
            context_file=Path(self.var_ctx.get()) if self.var_ctx.get().strip() else None,
            learn_descriptors=self.var_learn.get(),
            reset_context=self.var_reset.get(),
            ollama_url=self.var_ollama_url.get().strip(),
            ollama_model=self.var_ollama_model.get().strip(),
            clip_node_id=self.var_clip_node.get().strip(),
        )
        save_config(s)
        self.generate_button.config(state="disabled")
        threading.Thread(target=self._worker, args=(s,), daemon=True).start()

    def _worker(self, s: Settings) -> None:
        try:
            self.log_print("--- Starting Generation ---")
            if not s.input_pdf or not s.input_pdf.exists():
                messagebox.showerror("Error", "Please select a valid input PDF.")
                return

            if not s.workflow_file or not s.workflow_file.exists():
                messagebox.showerror("Error", "Please select a valid ComfyUI workflow JSON.")
                return

            ensure_dir(s.output_dir)

            bank = ContextBank(BANK_PATH)
            extractor = DescriptorExtractor()
            snapshot_path = s.context_file or (s.output_dir / "context_snapshot.md")

            if s.reset_context:
                bank.reset()
                if snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                    except OSError:
                        pass
                self.log_print("✓ Context has been reset.")

            pages = extract_pages(s.input_pdf)
            total = len(pages)
            page_numbers = parse_pages_spec(s.pages_range, total)
            self.log_print(
                f"✓ Loaded {total} pages from PDF. Processing {len(page_numbers)} pages: {s.pages_range or 'All'}")

            base_seed = int(time.time())

            for i in page_numbers:
                page_text = pages[i - 1]
                if not page_text.strip():
                    self.log_print(f"○ Skipping page {i} (no text found).")
                    continue

                self.log_print(f"\n--- Processing Page {i} ---")
                final_img_path = s.output_dir / f"page_{i:04d}.png"
                if s.skip_existing and final_img_path.exists():
                    self.log_print(f"⏭ Skipping page {i} (output file already exists).")
                    continue

                # Learn descriptors from this page's text
                if s.learn_descriptors:
                    try:
                        desc_pairs: List[tuple] = []
                        used_ollama_desc = False
                        # Use Ollama for descriptor extraction if a model is specified
                        if s.ollama_model:
                            self.log_print(
                                f"  > Extracting descriptors via Ollama (URL: {s.ollama_url}, model: {s.ollama_model})")
                            try:
                                pairs = ollama_extract_descriptors(s.ollama_model, page_text, s.ollama_url)
                                if pairs:
                                    used_ollama_desc = True
                                    self.log_print(f"  > Ollama returned {len(pairs)} descriptor(s)")
                                    for name, desc in pairs:
                                        self.log_print(f"    - {name}: {desc}")
                                        desc_pairs.append((name, desc, []))
                                else:
                                    self.log_print(
                                        "  > Ollama returned no descriptors; falling back to regex extractor.")
                            except Exception as e:
                                self.log_print(
                                    f"  > (!) Ollama descriptor extraction failed: {e}; falling back to regex extractor.")
                        # Fallback to regex extractor if nothing from Ollama or no model
                        if not desc_pairs:
                            regex_descs = extractor.suggest(page_text)
                            if regex_descs:
                                self.log_print(f"  > Regex extractor found {len(regex_descs)} descriptor(s)")
                                for d in regex_descs:
                                    self.log_print(f"    - {d.name}: {d.description}")
                                    desc_pairs.append((d.name, d.description, d.aliases))
                            else:
                                self.log_print("  > No new descriptors found on this page.")
                        # Upsert descriptors into the context bank
                        if desc_pairs:
                            for name, desc, aliases in desc_pairs:
                                bank.upsert(name, desc, aliases)
                            bank.save()
                            export_context_snapshot(bank, snapshot_path)
                    except Exception as e:
                        self.log_print(f"  > (!) Descriptor learning failed: {e}")

                # Build context snippet for summarisation
                context_snippet = bank.relevant_snippet(page_text)

                # Create the base prompt for the image
                base_prompt = ""
                used_ollama_prompt = False
                if s.summarise:
                    if s.openai_key:
                        self.log_print("  > Summarizing page with OpenAI…")
                        base_prompt = openai_summarise(s.openai_key, page_text[:4000], s.token_cap)
                    elif s.ollama_model:
                        self.log_print(f"  > Generating prompt with Llama via Ollama (model: {s.ollama_model})…")
                        try:
                            base_prompt = ollama_generate_prompt(s.ollama_model, context_snippet, page_text,
                                                                 s.ollama_url)
                            used_ollama_prompt = True if base_prompt.strip() else False
                        except Exception as e:
                            self.log_print(f"  > (!) Llama prompt generation failed: {e}")
                            base_prompt = ""

                if not base_prompt.strip():
                    # Fallback if summarization fails or is disabled
                    base_prompt = " ".join(re.sub(r"\s+", " ", page_text).split()[:120])
                    self.log_print("  > Using first part of page text as prompt.")

                # Determine final prompt: if Llama summarization was used and succeeded, skip enrichment
                if used_ollama_prompt and base_prompt.strip():
                    final_prompt = base_prompt.strip()
                else:
                    final_prompt = enrich_prompt_with_context(bank, base_prompt, page_text=page_text)

                self.log_print(f"  > Final Prompt: {final_prompt[:200]}{'...' if len(final_prompt) > 200 else ''}")

                #TODO: Format the final prompt to show context separate to the actual page summary

                # Write the full final prompt to a .txt file for this page (useful for debugging)
                try:
                    prompt_path = s.output_dir / f"prompt_{i:04d}.txt"
                    prompt_path.write_text(final_prompt, encoding="utf-8")
                except Exception as e:
                    # Do not halt execution if writing fails; just log the issue
                    self.log_print(f"  > (!) Could not write prompt file for page {i}: {e}")

                # Generate image
                try:
                    page_seed = base_seed + i
                    raw_img = comfy_inject_and_generate(
                        s.comfy_url,
                        s.workflow_file,
                        final_prompt,
                        s.output_dir,
                        page_seed,
                        self.log_print,
                        clip_node_id=s.clip_node_id or None,
                    )

                    if s.add_text_under_image:
                        self.log_print(f"  > Adding caption to {raw_img.name}...")
                        compose_with_caption(raw_img, page_text, final_img_path)
                        if raw_img.exists() and raw_img.resolve() != final_img_path.resolve():
                            try:
                                raw_img.unlink()
                            except OSError:
                                pass
                    else:
                        if raw_img.resolve() != final_img_path.resolve():
                            raw_img.replace(final_img_path)

                    self.log_print(f"✓ Page {i} generation complete: {final_img_path.name}")
                except Exception as e:
                    self.log_print(f"✖ Page {i} image generation failed: {e}")
                    traceback.print_exc()
                    continue

            self.log_print("\n--- All pages processed. ---")
        except Exception as e:
            self.log_print(f"\n--- FATAL ERROR --- \n{e}\n{traceback.format_exc()}")
            messagebox.showerror("Fatal Error", str(e))
        finally:
            self.generate_button.config(state="normal")


# ---------------- Config I/O ----------------

def save_config(s: Settings) -> None:
    data = {
        "input_pdf": str(s.input_pdf) if s.input_pdf else "",
        "output_dir": str(s.output_dir),
        "openai_key": s.openai_key,
        "summarise": s.summarise,
        "pages_range": s.pages_range,
        "skip_existing": s.skip_existing,
        "add_text_under_image": s.add_text_under_image,
        "comfy_url": s.comfy_url,
        "workflow_file": str(s.workflow_file) if s.workflow_file else "",
        "context_file": str(s.context_file) if s.context_file else "",
        "learn_descriptors": s.learn_descriptors,
        "reset_context": s.reset_context,
        "ollama_url": s.ollama_url,
        "ollama_model": s.ollama_model,
        "clip_node_id": s.clip_node_id,
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def load_config() -> Settings:
    s = Settings()
    if CONFIG_PATH.exists():
        try:
            d = json.loads(CONFIG_PATH.read_text())
            pdf_path = d.get("input_pdf")
            s.input_pdf = Path(pdf_path) if pdf_path else None
            s.output_dir = Path(d.get("output_dir", str(s.output_dir)))
            s.openai_key = d.get("openai_key", s.openai_key)
            s.summarise = bool(d.get("summarise", s.summarise))
            s.pages_range = d.get("pages_range", s.pages_range)
            s.skip_existing = bool(d.get("skip_existing", s.skip_existing))
            s.add_text_under_image = bool(d.get("add_text_under_image", s.add_text_under_image))
            s.comfy_url = d.get("comfy_url", s.comfy_url)
            wf_path = d.get("workflow_file")
            s.workflow_file = Path(wf_path) if wf_path else None
            cf_path = d.get("context_file")
            s.context_file = Path(cf_path) if cf_path else None
            s.learn_descriptors = bool(d.get("learn_descriptors", s.learn_descriptors))
            s.reset_context = bool(d.get("reset_context", s.reset_context))
            s.ollama_url = d.get("ollama_url", s.ollama_url)
            s.ollama_model = d.get("ollama_model", s.ollama_model)
            s.clip_node_id = d.get("clip_node_id", s.clip_node_id)
        except (json.JSONDecodeError, KeyError):
            pass  # Use defaults if config is malformed
    return s


if __name__ == "__main__":
    app = App()
    app.mainloop()
