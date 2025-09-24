# 📖 BookAI – Context-Aware Book Illustrator

BookAI turns an entire book (or any PDF) into a **collection of images**.  
It automatically generates page-by-page illustrations, while learning details about characters and settings to keep them **visually consistent across the story**.  

In other words: you give it a book → it gives you a gallery of illustrations, one per page, enriched by context-aware prompts.

---

## ✨ What It Does

- **Reads your book/PDF**: extracts text from each page.  
- **Learns context over time**: builds a "context bank" of characters and descriptors (e.g., *“Clary: red hair, freckles”*, *“Jace: golden eyes”*).  
- **Builds enriched prompts**: merges raw page text with context (e.g.,  
  `"Clary enters the room"` → `"Clary (a 15-year-old girl with red hair and freckles) enters the room"`).  
- **Generates illustrations**: injects prompts into your ComfyUI workflow, producing page-specific images.  
- **Outputs a gallery**: every run creates a collection of images, prompts, and snapshots.  

---

## 🖼 Example

Input (page 100, *City of Bones*):  
```
Clary enters the room.
```

Output prompt:  
```
Clary (a fifteen-year-old girl with red hair and freckles) enters the room.
```

Which yields a more **faithful illustration**, and the character will look the same in later pages.

---

## ⚙️ Setup

We recommend using a Python virtual environment.

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate    # or activate.fish

# Install dependencies
pip install -r requirements.txt
```

### Optional: spaCy NLP model
For better descriptor extraction, install the small English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

> Without this, BookAI will still work, but descriptor learning will rely on regex rules and Ollama LLMs.

---

## 🖥️ ComfyUI

Make sure your ComfyUI instance is running.  
Default URL: [http://127.0.0.1:8188](http://127.0.0.1:8188).  
You’ll need to point BookAI to a ComfyUI **workflow JSON** (the “API variant”).

---

## ▶️ How to Run

Start the Flask web interface:

```bash
python -m bookai.web.app
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.  
From there you can:  

- Configure PDF, output folder, API keys, and workflow.  
- Watch logs stream in real-time.  
- See images appear in the **per-run gallery**.  
- Browse all generated outputs in the **Gallery tab**.  

---

## 🔍 How It Works (High Level)

1. **PDF Parsing** – extracts raw text per page.  
2. **Context Learning** – updates `context_bank.json` with character descriptors using:
   - Regex / spaCy  
   - Ollama LLMs (for smarter extraction)  
3. **Prompt Generation** – combines page text + context, optionally summarizes with OpenAI or Ollama.  
4. **Image Generation** – injects prompt into ComfyUI workflow, queues run, downloads results.  
5. **Outputs** – a gallery of images, context snapshots, and prompts per run.  

---

## 📚 Documentation

See [docs/structure.md](docs/structure.md) for:  
- File-by-file breakdown  
- Folder responsibilities  
- Coding style guidelines  
- Detailed workflow and architecture diagram  

---

## 🚧 Status

- ✅ Core PDF → Context → Prompt → Image pipeline  
- ✅ Flask web UI with live logs and gallery  
- ✅ Ollama + OpenAI integration  
- 🚧 Extended prompt tuning  
- 🚧 More UI polish  

---

## 📝 License

MIT (for now — subject to change as project evolves)

---

## 🙋 About

BookAI is built by:  
- **Andrew** – Core developer, architecture, orchestration, and domain logic.  
- **Aaron** – Core developer, Image generation and prompt development.