# 🧠 DABIB – Definitive Artificial Biological Imitation Brain

**DABIB** (Definitive Artificial Biological Imitation Brain) is a FastAPI-based, memory‑aware proxy for Large Language Models (LLMs). It imitates a simplified biological brain by combining:

- Episodic memory (hippocampus‑like)
- Semantic memory (neocortex‑like)
- Procedural skills (habit / skill system)

The proxy wraps an OpenAI‑compatible LLM endpoint and automatically manages long‑term memory and context injection.

> Author: **Alberto Arce**, independent researcher.

---

## Key features

- **Episodic memory (hippocampus‑like)**  
  Stores raw user interactions as episodes with timestamps, session identifiers and importance scores.

- **Semantic memory (neocortex‑like)**  
  Consolidates atomic, long‑term facts about the user and the external world for durable knowledge.

- **Procedural skills**  
  Extracts reusable “how‑to” skills (multi‑step procedures) from interactions and injects them when relevant.

- **Sleep / consolidation cycles**  
  After a configurable number of interactions, DABIB:
  - Updates a narrative user profile.
  - Distinguishes personal vs external atomic facts.
  - Optionally verifies external facts via a SearxNG meta‑search instance.
  - Stores new facts and skills into ChromaDB and clears short‑term episodes.

- **OpenAI‑compatible chat endpoint**  
  Exposes `POST /v1/chat/completions`, so existing OpenAI‑style clients can talk to DABIB by just changing the base URL.

For a more formal motivation and design rationale, see  
`docs/dabib-motivation.md`.

---

## Project layout

```text
dabib/
├── app/
│   ├── __init__.py          # Package initializer
│   └── main.py              # FastAPI app and memory logic
├── docs/
│   ├── index.html           # Simple landing page (for GitHub Pages)
│   └── dabib-motivation.md  # Scientific-style article about DABIB
├── tests/
│   └── test_basic.py        # Placeholder unit test
├── .gitignore
├── LICENSE                  # License text (e.g. MIT)
├── README.md
└── requirements.txt
```

This structure follows common FastAPI and Python layout practices: code in an `app/` package, tests in `tests/`, docs in `docs/`, and a single `main.py` entrypoint.[web:42]

---

## Requirements

- Python 3.10+
- Local LLM server exposing an OpenAI‑compatible endpoint:
  - `POST /v1/chat/completions`
- Local ChromaDB (using the embedded persistent client)
- Optional but recommended: SearxNG instance for external fact verification

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running DABIB

From the project root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Main chat endpoint:

```text
POST http://localhost:8000/v1/chat/completions
```

Configure your OpenAI‑compatible client (e.g. LibreChat or a custom UI) to point to this URL instead of the original LLM server.

---

## High‑level architecture

- **FastAPI**  
  Hosts the HTTP API and streams responses back to the client.

- **ChromaDB**  
  Provides vector‑based storage for:
  - `episodic_memory` (short‑term conversation episodes)
  - `semantic_memory` (consolidated long‑term facts)
  - `procedural_skills` (how‑to blocks)

- **SearxNG (optional)**  
  Used to cheaply check a subset of external facts discovered during consolidation.

- **LLM backend**  
  Any OpenAI‑compatible chat completion server (e.g. a local LLaMA‑based server) used for:
  - Answer generation
  - Episodic memory compression
  - Sleep/consolidation reasoning

---

## Author and license

Created by **Alberto Arce**, independent researcher.  

License: MIT (or another OSI-approved license of your choice), see `LICENSE`.
