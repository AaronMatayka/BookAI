from __future__ import annotations

import re, time, traceback
from pathlib import Path
from typing import Callable, Optional, List

from src.config import Config, BANK_PATH, ensure_dir, parse_pages_spec
from src.ports.logger import Logger
from src.ports.pdf_extractor import PDFExtractor
from src.ports.context_store import ContextStore
from src.ports.descriptor_learner import DescriptorLearner
from src.ports.summarizer import Summarizer
from src.ports.prompt_builder import PromptBuilder
from src.ports.image_renderer import ImageRenderer, ImageComposer


class BookAIPipeline:
    """
    High-level orchestration. Knows nothing about pdfplumber/OpenAI/Comfy/Pillow.
    It only talks to *interfaces* (ports). Swapping implementations is trivial.
    """

    def __init__(
        self,
        cfg: Config,
        logger: Logger,
        pdf: PDFExtractor,
        ctx: ContextStore,
        descriptor_learners: List[DescriptorLearner],
        summarizer: Optional[Summarizer],
        prompt_builder: PromptBuilder,
        renderer: ImageRenderer,
        composer: Optional[ImageComposer] = None,
    ) -> None:
        self.cfg = cfg
        self.log = logger.log
        self.pdf = pdf
        self.ctx = ctx
        self.learners = descriptor_learners
        self.summarizer = summarizer
        self.prompts = prompt_builder
        self.renderer = renderer
        self.composer = composer

    def run(self, on_image: Optional[Callable[[Path], None]] = None) -> None:
        c = self.cfg

        # Basic input validation
        if not c.input_pdf or not c.input_pdf.exists():
            raise FileNotFoundError("Please provide a valid input PDF.")
        if not c.workflow_file or not c.workflow_file.exists():
            raise FileNotFoundError("Please provide a valid ComfyUI workflow JSON.")
        ensure_dir(c.output_dir)

        # Reset context if requested
        if c.reset_context:
            self.ctx.reset()
            if c.context_file and Path(c.context_file).exists():
                try: Path(c.context_file).unlink()
                except OSError: pass
            self.log("✓ Context has been reset.")

        # Read pages and select which to process
        pages = self.pdf.extract_pages(c.input_pdf)
        total = len(pages)
        page_numbers = parse_pages_spec(c.pages_range, total)
        self.log(f"✓ Loaded {total} pages from PDF. Processing {len(page_numbers)} pages: {c.pages_range or 'All'}")

        # Loop
        base_seed = int(time.time())
        for i in page_numbers:
            page_text = pages[i-1]
            if not page_text.strip():
                self.log(f"○ Skipping page {i} (no text).")
                continue

            self.log(f"\n--- Processing Page {i} ---")
            final_img = c.output_dir / f"page_{i:04d}.png"

            if c.skip_existing and final_img.exists():
                self.log(f"⏭ Skipping page {i} (already exists).")
                if on_image: on_image(final_img)
                continue

            # Learn descriptors
            if c.learn_descriptors:
                try:
                    added = 0
                    for learner in self.learners:
                        triples = learner.learn(page_text) or []
                        for name, desc, aliases in triples:
                            self.log(f"    - {name}: {desc}")
                            self.ctx.upsert(name, desc, aliases)
                            added += 1
                    if added:
                        self.ctx.save()
                        snapshot = str(c.context_file or (c.output_dir / "context_snapshot.md"))
                        self.ctx.export_snapshot(snapshot)
                        self.log(f"  > Context updated with {added} descriptor(s).")
                    else:
                        self.log("  > No new descriptors found.")
                except Exception as e:
                    self.log(f"  > (!) Descriptor learning failed: {e}")

            # Summarize or fallback to trimmed text
            base_prompt = ""
            used_summarizer = False
            if c.summarise and self.summarizer:
                try:
                    base_prompt = self.summarizer.summarize(page_text[:4000], c.token_cap).strip()
                    used_summarizer = bool(base_prompt)
                except Exception as e:
                    self.log(f"  > (!) Summarizer failed: {e}")

            if not base_prompt:
                base_prompt = " ".join(re.sub(r"\s+"," ",page_text).split()[:120])
                self.log("  > Using first part of page text as prompt.")

            # Enrich with context unless summarizer already gave a final prompt (design choice)
            final_prompt = base_prompt if used_summarizer else self.prompts.build(base_prompt, page_text)
            self.log(f"  > Final Prompt: {final_prompt[:200]}{'...' if len(final_prompt)>200 else ''}")

            # Persist prompt
            try:
                (c.output_dir / f"prompt_{i:04d}.txt").write_text(final_prompt, encoding="utf-8")
            except Exception as e:
                self.log(f"  > (!) Could not write prompt file for page {i}: {e}")

            # Render
            try:
                seed = base_seed + i
                tmp = self.renderer.render(
                    workflow_file=c.workflow_file,
                    prompt_text=final_prompt,
                    out_dir=c.output_dir,
                    seed=seed,
                    clip_node_id=c.clip_node_id or None,
                    progress=self.log,
                )
                if c.add_text_under_image and self.composer:
                    self.log(f"  > Adding caption to {tmp.name}…")
                    self.composer.compose_with_caption(tmp, page_text, final_img)
                    if tmp.exists() and tmp.resolve() != final_img.resolve():
                        try: tmp.unlink()
                        except OSError: pass
                else:
                    if tmp.resolve() != final_img.resolve():
                        tmp.replace(final_img)
                self.log(f"✓ Page {i} complete: {final_img.name}")
                if on_image: on_image(final_img)
            except Exception as e:
                self.log(f"✖ Page {i} failed: {e}")
                traceback.print_exc()
                continue

        self.log("\n--- All pages processed. ---")
