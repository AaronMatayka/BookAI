#!/usr/bin/env python3
from __future__ import annotations

import threading
from pathlib import Path
from typing import List
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.config import Config, ConfigManager, ensure_dir
from src.application.pipeline import BookAIPipeline

# Adapters
from src.adapters.stdout_logger import StdoutLogger
from src.adapters.pdfplumber_extractor import PdfPlumberExtractor
from src.adapters.context_bank_store import ContextBankStore
from src.adapters.regex_descriptor_learner import RegexDescriptorLearner
from src.adapters.ollama_descriptor_learner import OllamaDescriptorLearner
from src.adapters.openai_summarizer import OpenAISummarizer
from src.adapters.ollama_summarizer import OllamaSummarizer
from src.adapters.prompt_enricher import PromptEnricher
from src.adapters.comfy_client import ComfyClient
from src.adapters.pillow_image_renderer import PillowImageComposer

# State for current run (UI polls this)
LOGS: List[str] = []
FILES: List[str] = []
RUNNING = False

def ui_log(line: str) -> None:
    LOGS.append(line)

def on_image(path: Path) -> None:
    FILES.append(path.name)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str((Path.home() / ".config" / "src" / "uploads").absolute())
ensure_dir(Path(app.config["UPLOAD_FOLDER"]))

@app.get("/")
def index():
    cfg = ConfigManager.load()
    return render_template("index.html", s=cfg)

@app.post("/start")
def start():
    global RUNNING, LOGS, FILES
    if RUNNING:
        return jsonify({"ok": False, "error": "A run is already in progress."}), 409

    # Collect uploads (optional)
    input_pdf = request.files.get("input_pdf")
    workflow_file = request.files.get("workflow_file")
    uploaded_pdf = None
    uploaded_wf = None
    if input_pdf and input_pdf.filename:
        uploaded_pdf = Path(app.config["UPLOAD_FOLDER"]) / input_pdf.filename
        input_pdf.save(uploaded_pdf)
    if workflow_file and workflow_file.filename:
        uploaded_wf = Path(app.config["UPLOAD_FOLDER"]) / workflow_file.filename
        workflow_file.save(uploaded_wf)

    # Load config and map form fields
    cfg = ConfigManager.load()
    if uploaded_pdf: cfg.input_pdf = uploaded_pdf
    if uploaded_wf: cfg.workflow_file = uploaded_wf

    def b(name: str, default: bool) -> bool:
        return request.form.get(name, "off") == "on" if name in request.form else default

    cfg.output_dir = Path(request.form.get("output_dir", str(cfg.output_dir)).strip() or str(cfg.output_dir))
    cfg.context_file = Path(request.form.get("context_file", str(cfg.context_file or "")) or str(cfg.output_dir / "context_snapshot.md"))
    cfg.pages_range = request.form.get("pages_range", cfg.pages_range).strip()
    cfg.comfy_url = request.form.get("comfy_url", cfg.comfy_url).strip()
    cfg.openai_key = request.form.get("openai_key", cfg.openai_key).strip()
    cfg.ollama_url = request.form.get("ollama_url", cfg.ollama_url).strip()
    cfg.ollama_model = request.form.get("ollama_model", cfg.ollama_model).strip()
    cfg.clip_node_id = request.form.get("clip_node_id", cfg.clip_node_id).strip()
    cfg.summarise = b("summarise", cfg.summarise)
    cfg.learn_descriptors = b("learn_descriptors", cfg.learn_descriptors)
    cfg.reset_context = b("reset_context", cfg.reset_context)
    cfg.skip_existing = b("skip_existing", cfg.skip_existing)
    cfg.add_text_under_image = b("add_text_under_image", cfg.add_text_under_image)

    ConfigManager.save(cfg)

    # Build pipeline (compose adapters)
    ctx_store = ContextBankStore(bank_path=Path.home() / ".config" / "src" / "context_bank.json")
    logger = StdoutLogger(sink=ui_log)
    pdf = PdfPlumberExtractor()
    learners = []
    if cfg.ollama_model:
        learners.append(OllamaDescriptorLearner(model=cfg.ollama_model, url=cfg.ollama_url))
    learners.append(RegexDescriptorLearner())  # fallback/extra

    summarizer = None
    if cfg.summarise:
        if cfg.openai_key:
            summarizer = OpenAISummarizer(api_key=cfg.openai_key)
        elif cfg.ollama_model:
            summarizer = OllamaSummarizer(model=cfg.ollama_model, url=cfg.ollama_url,
                                          context_supplier=ctx_store.relevant_snippet)

    prompt_builder = PromptEnricher(context_store=ctx_store)
    renderer = ComfyClient(base_url=cfg.comfy_url)
    composer = PillowImageComposer()

    pipeline = BookAIPipeline(
        cfg=cfg,
        logger=logger,
        pdf=pdf,
        ctx=ctx_store,
        descriptor_learners=learners,
        summarizer=summarizer,
        prompt_builder=prompt_builder,
        renderer=renderer,
        composer=composer,
    )

    LOGS = ["--- Starting Generation ---"]
    FILES = []
    RUNNING = True

    def worker():
        global RUNNING
        try:
            pipeline.run(on_image=on_image)
            LOGS.append("\n--- Run finished. ---")
        except Exception as e:
            LOGS.append(f"\n--- FATAL ERROR ---\n{e}")
        finally:
            RUNNING = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})

@app.get("/logs")
def logs():
    return jsonify({"running": RUNNING, "lines": LOGS})

@app.get("/outputs")
def outputs():
    return jsonify({"running": RUNNING, "files": FILES})

@app.get("/all-outputs")
def all_outputs():
    cfg = ConfigManager.load()
    outdir = cfg.output_dir
    if not outdir.exists():
        return jsonify({"files": []})
    files = [p.name for p in outdir.iterdir() if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg",".webp")]
    files.sort()
    return jsonify({"files": files})

@app.get("/download/<path:name>")
def download(name: str):
    cfg = ConfigManager.load()
    return send_from_directory(cfg.output_dir, name, as_attachment=False)

if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
