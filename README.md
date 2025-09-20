BookAI - AI Book Illustrator

This application reads a book, or any PDF, extracts character and location descriptions per page to build a "context bank," and then generates images for each page using ComfyUI or online image generators (WIP)


The goal is to create context-aware illustrations. As the program reads the book, it learns details (e.g., "Clary has red hair," "Jace has gold eyes"). For later pages, it injects this learned context into the image prompt, ensuring characters are visually consistent.

For example, a simple prompt from page 100 of the book "City of Bones", like "Clary enters the room" might become "Clary (a fifteen-year-old girl with red hair and freckles) enters the room," leading to a much more accurate image.
Setup

    Dependencies: Install the required Python libraries. It's highly recommended to use a virtual environment.
    To create venv:
    python3 -m venv venv 
    source .venv/bin/activate or activate.fish
    
    pip install -r requirements.txt

    NLP Model (Recommended): For the best descriptor extraction results, the program uses the spaCy library. You'll need to download its small English model. (WIP, currently working on using an LLM through Ollama for context-aware information generation, as I was struggling with using local options like regex and spaCy for the project.) 

    python -m spacy download en_core_web_sm

    (The program will still function without this, but the context-learning will be less effective.)

    ComfyUI:

        Make sure your ComfyUI instance is running. The default URL is http://127.0.0.1:8188, but you can change this in the app.

        In the BookAI application, you'll need to point to a ComfyUI workflow file of the API varient.

How to Run

Simply execute the main Python script:

python bookai_application.py

Fill in the fields in the user interface and click "Generate".
How It Works

    PDF Parsing: The app reads the text from each page of your selected PDF.

    ENTIRE FILE WIP:
    Context Learning (context_bank.py):

        For each page, the DescriptorExtractor looks for named entities (people, places).

        It uses advanced logic (spaCy or regex) to find visual descriptions attached to those entities, including handling appositives ("Clary, a girl with red hair..."), possessives ("The boy's eyes were green..."), and simple coreferences ("A boy walked in. He had blue hair...").

        This information is saved to context_bank.json in your config folder. The descriptions become richer as the app reads more of the book.

    Prompt Generation:

        It takes the raw text of the current page.

        Optionally (if an OpenAI key is provided), it uses GPT-4o-mini to summarize the page into a concise, visual scene description (untested). If not, it uses the first chunk of text from the page.

        It then enriches this summary by injecting the relevant learned context for any characters mentioned on that specific page.

    Image Generation (bookai_application.py):

        It connects to your ComfyUI API.

        It dynamically injects the final, context-rich prompt into the text nodes of your chosen workflow JSON.

        It sets a unique, deterministic seed for each page number.

        It queues the prompt, waits for the image to be generated, and downloads it to your output folder.
    Ollama Integration (ollama_connector.py) 

        It connects to Ollama, by default on 127.0.0.1:11434.

        It then will use the user-specified LLM to generate both the context-important information for the context.txt file, such as Clary has red hair, and the final prompt for the image generator, by taking the page of the book as input, injecting relevent info from the context.txt file, and injecting this to ComfyUI for the image generation pipeline.
