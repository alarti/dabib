# app/main.py
"""
DABIB – Definitive Artificial Biological Imitation Brain

A FastAPI-based, memory-aware proxy for Large Language Models (LLMs) that imitates
a simplified biological brain using:
- episodic memory (hippocampus-like),
- semantic memory (neocortex-like),
- procedural skills.

Author: Alberto Arce (independent researcher).
"""

import os
import json
import uuid
import random
import traceback
from datetime import datetime

import requests
import chromadb
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="DABIB – Definitive Artificial Biological Imitation Brain",
    description="Biological-inspired memory system for LLMs with episodic, semantic and procedural memory.",
    version="0.2.0",
)

# ---------------------------------------------------------------------
# Configuration: LLM, memory and external verification
# ---------------------------------------------------------------------

LLM_SERVER_URL = "http://127.0.0.1:8090/v1/chat/completions"

MEMORY_DB_PATH = "./biological_memory"
USER_PROFILE_FILE = "user_profile.txt"
EPISODE_LOG_FILE = "episodes_log.jsonl"

EPISODIC_COLLECTION_NAME = "episodic_memory"
SEMANTIC_COLLECTION_NAME = "semantic_memory"
SKILL_COLLECTION_NAME = "procedural_skills"

INTERACTIONS_BEFORE_SLEEP = 5
MAX_EPISODES_PER_SLEEP = 50
MAX_REMOTE_FACTS_FOR_REPLAY = 10
N_EPISODIC_CONTEXT = 2
N_SEMANTIC_CONTEXT = 3
N_SKILLS_CONTEXT = 1

# Episodic memory compression thresholds
MAX_EPISODES_IN_HIPPOCAMPUS = 200
COMPRESSION_WINDOW = 20

# SearxNG configuration (external verification)
SEARXNG_URL = "http://127.0.0.1:8081/search"
SEARXNG_TIMEOUT = 15
MAX_SOURCES_PER_FACT = 3
MAX_EXTERNAL_FACTS_TO_VERIFY = 3

# Runtime state
interactions_since_last_sleep = 0

# Session state: { session_id: { "user_name": str | None, "last_interaction": str | None, "last_sleep": str | None } }
SESSIONS: dict[str, dict] = {}

# ---------------------------------------------------------------------
# Memory initialization (ChromaDB collections)
# ---------------------------------------------------------------------

db_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
episodic_memory = db_client.get_or_create_collection(name=EPISODIC_COLLECTION_NAME)
semantic_memory = db_client.get_or_create_collection(name=SEMANTIC_COLLECTION_NAME)
procedural_skills = db_client.get_or_create_collection(name=SKILL_COLLECTION_NAME)

print("🧠 DABIB central nervous system initialized and waiting for connections...")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def current_iso_timestamp() -> str:
    """Return current UTC datetime as ISO‑8601 string."""
    return datetime.utcnow().isoformat()


def read_user_profile() -> str:
    """Read the narrative user profile, or a default message if not present."""
    if os.path.exists(USER_PROFILE_FILE):
        try:
            with open(USER_PROFILE_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "No consolidated user profile is available yet."


def write_user_profile(text: str) -> None:
    """Persist the narrative user profile to disk."""
    os.makedirs(os.path.dirname(USER_PROFILE_FILE) or ".", exist_ok=True)
    with open(USER_PROFILE_FILE, "w", encoding="utf-8") as f:
        f.write(text.strip())


def append_episode_log(episodes: list[dict]) -> None:
    """Append episodic records to a JSONL log for auditing and analysis."""
    if not episodes:
        return
    with open(EPISODE_LOG_FILE, "a", encoding="utf-8") as f:
        for ep in episodes:
            record = {
                "id": ep["id"],
                "document": ep["document"],
                "metadata": ep.get("metadata", {}),
                "logged_at": current_iso_timestamp(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Session and user name handling
# ---------------------------------------------------------------------


def extract_session_id(request: Request, body: dict) -> str:
    """Extract a session identifier from query params or body, or derive a default."""
    query_params = request.query_params

    session_id = (
        query_params.get("session_id")
        or query_params.get("conversation_id")
        or query_params.get("user_id")
        or body.get("session_id")
        or body.get("conversation_id")
        or body.get("user_id")
    )

    if not session_id:
        client = request.client
        session_id = (client.host if client else "anon") + ":default"

    return str(session_id)


def extract_user_name_from_payload(body: dict, last_message: dict, request: Request) -> str | None:
    """Try to infer the human user's name from different fields."""
    # 1) OpenAI-style "name" field in the last message
    name = last_message.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    # 2) "user" field in the body
    body_user = body.get("user")
    if isinstance(body_user, str) and body_user.strip():
        return body_user.strip()

    # 3) Explicit query parameter (e.g. mapped from a UI)
    query_params = request.query_params
    qp_name = query_params.get("user_name")
    if qp_name and qp_name.strip():
        return qp_name.strip()

    return None


def get_session_state(request: Request, body: dict, last_message: dict) -> tuple[str, dict]:
    """Return (session_id, state) and update user name and timestamps."""
    session_id = extract_session_id(request, body)
    state = SESSIONS.setdefault(
        session_id,
        {"user_name": None, "last_interaction": None, "last_sleep": None},
    )

    detected_name = extract_user_name_from_payload(body, last_message, request)
    if detected_name:
        state["user_name"] = detected_name

    state["last_interaction"] = current_iso_timestamp()
    return session_id, state


# ---------------------------------------------------------------------
# SearxNG: search and fact verification
# ---------------------------------------------------------------------


def search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """Query SearxNG and return a list of results: [{title, url, snippet}, ...]."""
    try:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "pageno": 1,
            "categories": "general",
        }
        response = requests.get(SEARXNG_URL, params=params, timeout=SEARXNG_TIMEOUT)
        if response.status_code != 200:
            print(f"[⚠️ SEARXNG] Status {response.status_code} while searching '{query[:60]}'")
            return []

        data = response.json()
        raw_results = data.get("results", []) or []
        results = []
        for item in raw_results[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", "") or item.get("content", ""),
                }
            )
        return results
    except Exception as exc:
        print(f"[🛑 SEARXNG] Error while searching '{query[:60]}': {exc}")
        return []


def verify_fact_with_searxng(fact_text: str) -> dict:
    """
    Cheap external fact verification:
    - If SearxNG yields results → 'supported' and store URLs/snippets.
    - Otherwise → 'uncertain'.
    """
    results = search_searxng(fact_text, max_results=MAX_SOURCES_PER_FACT)
    if not results:
        return {
            "status": "uncertain",
            "sources": [],
        }

    sources: list[dict] = []
    for r in results[:MAX_SOURCES_PER_FACT]:
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        if url:
            sources.append(
                {
                    "url": url,
                    "snippet": snippet[:300],
                }
            )

    return {
        "status": "supported",
        "sources": sources,
    }


# ---------------------------------------------------------------------
# Episodic memory compression
# ---------------------------------------------------------------------


def compress_hippocampus() -> None:
    """
    When episodic memory grows too large, compress low-importance episodes
    into a short summary and re-insert it as a higher-importance document.
    """
    try:
        results = episodic_memory.get()
        docs = results.get("documents", []) or []
        ids = results.get("ids", []) or []
        metas = results.get("metadatas", []) or []

        total = len(docs)
        if total <= MAX_EPISODES_IN_HIPPOCAMPUS:
            return

        print(f"[🧹 COMPRESSION] Hippocampus at {total} episodes. Starting compaction...")

        episodes: list[dict] = []
        for doc, doc_id, meta in zip(docs, ids, metas):
            meta = meta or {}
            ts = meta.get("timestamp") or ""
            importance = meta.get("importance", 1)
            episodes.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "timestamp": ts,
                    "importance": importance,
                }
            )

        # Compress the least important and oldest episodes first
        episodes.sort(key=lambda x: (x["importance"], x["timestamp"]))
        batch = episodes[:COMPRESSION_WINDOW]
        if not batch:
            return

        batch_text = "\n- ".join(e["document"] for e in batch)

        compression_prompt = f"""You are compressing episodic memory for a user.

EPISODES:
- {batch_text}

TASK:
Write a very concise summary (3–5 sentences) capturing the essential information
from these episodes, without losing details that might be useful for future
conversations. Avoid literal repetition.
"""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a memory compressor. Generate faithful, very concise summaries.",
                },
                {"role": "user", "content": compression_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }

        print("[⚙️ COMPRESSION] Distilling batch of episodes...")
        response = requests.post(LLM_SERVER_URL, json=payload, timeout=60)

        if response.status_code != 200:
            print(f"[❌ COMPRESSION] Error while summarizing batch: {response.status_code}")
            return

        summary = response.json()["choices"][0]["message"]["content"].strip()
        if not summary:
            print("[⚠️ COMPRESSION] Empty summary, hippocampus not modified.")
            return

        episodic_memory.add(
            documents=[summary],
            ids=[str(uuid.uuid4())],
            metadatas=[
                {
                    "timestamp": current_iso_timestamp(),
                    "role": "system",
                    "origin": "compressed_summary",
                    "type": "summary",
                    "importance": 2,
                }
            ],
        )

        ids_to_delete = [e["id"] for e in batch]
        episodic_memory.delete(ids=ids_to_delete)

        print(f"[✅ COMPRESSION] Compressed {len(batch)} episodes into 1 summary.")

    except Exception as exc:
        print(f"[🛑 COMPRESSION ERROR] {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------
# Sleep / consolidation
# ---------------------------------------------------------------------


def run_sleep_consolidation() -> None:
    """
    Sleep phase:
    - Mix recent episodes with older semantic facts.
    - Update the narrative user profile.
    - Generate atomic personal and external facts.
    - Verify a subset of external facts via SearxNG.
    - Detect and store procedural skills.
    """
    global interactions_since_last_sleep
    print("\n[🌙 SLEEP PHASE] Starting DABIB consolidation...")

    try:
        old_profile = read_user_profile()

        episodic_results = episodic_memory.get()
        docs_epi = episodic_results.get("documents", []) or []
        ids_epi = episodic_results.get("ids", []) or []
        metas_epi = episodic_results.get("metadatas", []) or []

        if not docs_epi:
            print("[📭 SLEEP] No new episodes to consolidate.")
            return

        episodes: list[dict] = []
        for doc, doc_id, meta in zip(docs_epi, ids_epi, metas_epi):
            meta = meta or {}
            ts = meta.get("timestamp") or ""
            episodes.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "timestamp": ts,
                }
            )
        episodes.sort(key=lambda x: x["timestamp"])
        recent_episodes = episodes[-MAX_EPISODES_PER_SLEEP:]

        raw_memory = "\n- ".join(ep["document"] for ep in recent_episodes)

        semantic_results = semantic_memory.get()
        docs_sem = semantic_results.get("documents", []) or []

        remote_facts: list[str] = []
        if docs_sem:
            indices = list(range(len(docs_sem)))
            random.shuffle(indices)
            indices = indices[:MAX_REMOTE_FACTS_FOR_REPLAY]
            remote_facts = [docs_sem[i] for i in indices]

        remote_facts_text = "\n- ".join(remote_facts) if remote_facts else "None"

        consolidation_prompt = f"""You act as the neocortex of a biological-inspired brain for an AI system.
Your task is to update a stable user profile and to generate atomic long-term facts
and procedural skills.

CURRENT USER PROFILE (narrative, third person):
\"\"\"{old_profile}\"\"\"

REMOTE CONSOLIDATED FRAGMENTS (old semantic memory):
- {remote_facts_text}

NEW RECENT EPISODES TO INTEGRATE (episodic memory):
- {raw_memory}

OUTPUT FORMAT:
1) First, write a SINGLE updated profile in third person, concise but complete.
2) Then write a line exactly: ---PERSONAL_FACTS---
3) Then write a list of ATOMIC PERSONAL facts about the user (preferences, habits,
   biography, opinions), each line starting with "- ".
4) Then write a line exactly: ---EXTERNAL_FACTS---
5) Then write a list of ATOMIC EXTERNAL facts about the world (objective data,
   definitions, historical facts), each line starting with "- ".
6) Then write a line exactly: ---SKILLS---
7) Then optionally write one or more procedural skills using this format:

TITLE: short descriptive title
WHEN_TO_USE: when this skill should be applied
STEPS:
1) ...
2) ...
3) ...
---

Do not invent data that cannot be reasonably inferred from the context above.
"""

        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert memory condenser."},
                {"role": "user", "content": consolidation_prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }

        print("   [⚙️ SLEEP] Distilling information in the neocortex...")
        response = requests.post(LLM_SERVER_URL, json=payload, timeout=120)

        if response.status_code != 200:
            print(f"[❌ SLEEP ERROR] LLM inference failed: {response.status_code}")
            return

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Parse: profile + personal facts + external facts + skills
        personal_split = content.split("---PERSONAL_FACTS---")
        new_profile = personal_split[0].strip()
        personal_rest = personal_split[1] if len(personal_split) > 1 else ""

        external_split = personal_rest.split("---EXTERNAL_FACTS---")
        personal_text = external_split[0].strip() if len(external_split) > 0 else ""
        external_rest = external_split[1] if len(external_split) > 1 else ""

        skills_split = external_rest.split("---SKILLS---")
        external_text = skills_split[0].strip() if len(skills_split) > 0 else ""
        skills_text = skills_split[1].strip() if len(skills_split) > 1 else ""

        # Update narrative profile
        write_user_profile(new_profile)

        consolidation_run_id = str(uuid.uuid4())

        personal_facts = [
            line.strip()[2:].strip()
            for line in personal_text.splitlines()
            if line.strip().startswith("-")
        ]

        external_facts = [
            line.strip()[2:].strip()
            for line in external_text.splitlines()
            if line.strip().startswith("-")
        ]

        docs_to_store: list[str] = []
        metas_to_store: list[dict] = []

        # Personal facts (no external verification)
        for fact in personal_facts:
            docs_to_store.append(fact)
            metas_to_store.append(
                {
                    "type": "consolidated_fact",
                    "class": "personal",
                    "timestamp": current_iso_timestamp(),
                    "consolidation_run_id": consolidation_run_id,
                    "verification_status": "personal",
                    "sources": [],
                }
            )

        # External facts (optionally verified)
        if external_facts:
            print(f"[🔎 SLEEP] Verifying up to {MAX_EXTERNAL_FACTS_TO_VERIFY} external facts via SearxNG...")

        for idx, fact in enumerate(external_facts):
            if idx < MAX_EXTERNAL_FACTS_TO_VERIFY:
                verification = verify_fact_with_searxng(fact)
                status = verification["status"]
                sources = verification["sources"]
            else:
                status = "not_verified"
                sources = []

            docs_to_store.append(fact)
            metas_to_store.append(
                {
                    "type": "consolidated_fact",
                    "class": "external",
                    "timestamp": current_iso_timestamp(),
                    "consolidation_run_id": consolidation_run_id,
                    "verification_status": status,
                    "sources": sources,
                }
            )

        if docs_to_store:
            semantic_memory.add(
                documents=docs_to_store,
                ids=[str(uuid.uuid4()) for _ in docs_to_store],
                metadatas=metas_to_store,
            )

        # Procedural skills
        if skills_text:
            blocks = [b.strip() for b in skills_text.split("---") if b.strip()]
            skills_docs: list[str] = []
            skills_metas: list[dict] = []
            for block in blocks:
                skills_docs.append(block)
                skills_metas.append(
                    {
                        "type": "procedural_skill",
                        "timestamp": current_iso_timestamp(),
                        "consolidation_run_id": consolidation_run_id,
                        "usage_count": 0,
                    }
                )

            if skills_docs:
                procedural_skills.add(
                    documents=skills_docs,
                    ids=[str(uuid.uuid4()) for _ in skills_docs],
                    metadatas=skills_metas,
                )
                print(f"[🛠️ SKILLS] Stored {len(skills_docs)} new procedural skills.")

        # Log and clear episodic memory that has been consolidated
        append_episode_log(recent_episodes)
        episodic_memory.delete(ids=[ep["id"] for ep in recent_episodes])

        print("[✅ SLEEP COMPLETED] User identity updated and short-term memory cleaned.")
        print(f"---\nNEW PROFILE:\n{new_profile}\n---")
        interactions_since_last_sleep = 0

    except Exception as exc:
        print(f"[🛑 CRITICAL SLEEP ERROR] {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, background_tasks: BackgroundTasks):
    """
    OpenAI-compatible chat endpoint that:
    - stores user messages as episodic memory,
    - triggers sleep/consolidation when needed,
    - retrieves episodic, semantic and skill context,
    - enriches the last user message with memory and forwards it to the LLM server.
    """
    global interactions_since_last_sleep

    try:
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            return StreamingResponse(
                iter(['data: {"error": "No messages"}\n\n']),
                media_type="text/event-stream",
            )

        last = messages[-1]
        last_content = last.get("content", "")
        last_role = last.get("role", "user")

        session_id, session_state = get_session_state(request, body, last)
        user_name = session_state.get("user_name")
        has_known_name = bool(user_name)

        # Filter out noise or automatic messages
        is_noise = (
            "title for the conversation" in last_content
            or len(last_content.strip()) < 4
            or last_role != "user"
        )

        # Store episodic memory
        if not is_noise:
            episode_id = str(uuid.uuid4())
            metadata = {
                "timestamp": current_iso_timestamp(),
                "role": last_role,
                "origin": "user",
                "session_id": session_id,
                "user_name": user_name or "UNKNOWN",
                "type": "conversation",
                "importance": 1,
            }
            episodic_memory.add(
                documents=[last_content],
                ids=[episode_id],
                metadatas=[metadata],
            )
            interactions_since_last_sleep += 1
            print(
                f"\n[📥 EPISODIC LEARNING] Episode stored ({episode_id}). "
                f"Session: {session_id} User: {user_name or 'UNKNOWN'}. "
                f"Fatigue: {interactions_since_last_sleep}/{INTERACTIONS_BEFORE_SLEEP}"
            )

            # Trigger consolidation when cognitive fatigue threshold is reached
            if interactions_since_last_sleep >= INTERACTIONS_BEFORE_SLEEP:
                background_tasks.add_task(run_sleep_consolidation)

            # Always allow hippocampus compression to run opportunistically
            background_tasks.add_task(compress_hippocampus)
        else:
            print("\n[🧹 FILTER] Automatic or low-signal message ignored.")

        profile_text = read_user_profile()

        episodic_context: list[str] = []
        semantic_context: list[str] = []
        skill_context: list[str] = []

        # Retrieve relevant context from all memory systems
        if not is_noise:
            try:
                episodic_query = episodic_memory.query(
                    query_texts=[last_content],
                    n_results=N_EPISODIC_CONTEXT,
                )
                episodic_docs = episodic_query.get("documents", [[]])[0] or []
                episodic_context = episodic_docs

                semantic_query = semantic_memory.query(
                    query_texts=[last_content],
                    n_results=N_SEMANTIC_CONTEXT,
                )
                semantic_docs = semantic_query.get("documents", [[]])[0] or []
                semantic_context = semantic_docs

                skills_query = procedural_skills.query(
                    query_texts=[last_content],
                    n_results=N_SKILLS_CONTEXT,
                )
                skills_docs = skills_query.get("documents", [[]])[0] or []
                skill_context = skills_docs

                if episodic_context:
                    print(
                        "[🔍 HIPPOCAMPUS] Injected episodic memories: "
                        + " | ".join(c[:60] for c in episodic_context)
                    )
                if semantic_context:
                    print(
                        "[🧠 NEOCORTEX] Injected semantic facts: "
                        + " | ".join(c[:60] for c in semantic_context)
                    )
                if skill_context:
                    print(
                        "[🛠️ SKILLS] Injected skills: "
                        + " | ".join(c.splitlines()[0][:60] for c in skill_context)
                    )
            except Exception:
                traceback.print_exc()

        # Build the enriched prompt
        if not is_noise:
            episodic_block = "- " + "\n- ".join(episodic_context) if episodic_context else "None relevant"
            semantic_block = "- " + "\n- ".join(semantic_context) if semantic_context else "None relevant"
            skills_block = "\n\n".join(skill_context) if skill_context else "No specific skill found"

            name_str = user_name or "UNKNOWN"
            name_instruction = ""
            if not has_known_name:
                name_instruction = (
                    "\n\nIMPORTANT: You do not know the person's name yet. "
                    "Before answering anything else, politely ask how they would like "
                    "to be called. After they answer, also respond to their original message."
                )

            enriched_prompt = f"""### SYSTEM MEMORY (DABIB) ###
User identity (stable narrative profile):
{profile_text}

Known user name for this session ({session_id}):
{name_str}{name_instruction}

Long-term semantic facts:
{semantic_block}

Relevant episodic memories:
{episodic_block}

Procedural skills potentially useful for this request:
{skills_block}

### USER MESSAGE ###
{last_content}
"""
            messages[-1]["content"] = enriched_prompt
            body["messages"] = messages

        async def stream_generator():
            print("[🚀 INFERENCE] Forwarding request to LLM server...")
            with requests.post(LLM_SERVER_URL, json=body, stream=True) as r:
                for line in r.iter_lines(chunk_size=1):
                    if line:
                        yield line.decode("utf-8") + "\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

    except Exception:
        print("[🛑 PROXY ERROR] Unexpected failure in DABIB proxy.")
        traceback.print_exc()
        return StreamingResponse(
            iter(['data: {"error": "Internal DABIB error"}\n\n']),
            media_type="text/event-stream",
        )


# ---------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
