# DABIB: A Biological-Inspired Memory Architecture for Large Language Models

**Author:** Alberto Arce (independent researcher)  
**Status:** Technical report / working paper

---

## Abstract

Large Language Models (LLMs) are powerful sequence predictors but lack an
explicit, structured long-term memory comparable to biological brains.
Most deployments rely on short prompts or stateless APIs, which forces
clients to re-send context and limits personalization.

This report presents **DABIB** (Definitive Artificial Biological
Imitation Brain), a FastAPI-based middleware that wraps an
OpenAI-compatible LLM with three complementary memory subsystems:
episodic memory, semantic memory and procedural skills. Episodic
memories are collected as raw conversation episodes, periodically
compressed and consolidated into semantic facts and reusable skills
through a “sleep” phase. A lightweight external verification mechanism
based on SearxNG can check a subset of world facts before committing
them to long-term storage.

We describe the architecture, implementation details and potential
applications of DABIB as an experimental framework for studying
long-lived, personalized LLM agents.

---

## 1. Introduction

LLMs exhibit impressive performance on a wide range of tasks, yet
standard deployments typically treat them as stateless functions:
clients send a prompt and receive a response, with no persistent memory
beyond what is explicitly included in the request. This stands in
contrast to biological brains, where episodic, semantic and procedural
memory systems cooperate to support learning and behavior over long
time scales.

A growing body of work explores memory-augmented language models and
retrieval-augmented generation, but many implementations focus on
document retrieval rather than modelling autobiographical and procedural
memory at the level of individual users. DABIB aims to complement these
approaches with a biologically-inspired decomposition of memory and
simple mechanisms for consolidation and verification.

The main goals of DABIB are:

- To provide a practical, hackable implementation of a brain-like
  memory architecture for LLMs.
- To separate short-term episodic traces from consolidated semantic
  knowledge.
- To model procedural skills as explicit, reusable artifacts that can be
  retrieved independently of raw episodes.

---

## 2. Methods and system design

### 2.1 Overview

DABIB is implemented as a FastAPI service that exposes an
OpenAI-compatible `POST /v1/chat/completions` endpoint. Instead of
talking directly to the LLM server, clients send their requests to
DABIB. The middleware performs four main steps:

1. **Episodic capture** – each user message is stored as an episode
   in a ChromaDB collection, along with metadata such as timestamp,
   session ID and user name.
2. **Context retrieval** – for each new request, DABIB queries three
   collections (episodic, semantic and procedural) and injects the most
   relevant items into the prompt.
3. **LLM inference** – the enriched prompt is forwarded to an underlying
   LLM server that implements the OpenAI chat completions API.
4. **Sleep / consolidation** – after a configurable number of
   interactions, a background task merges recent episodes and selected
   semantic facts into atomic long-term facts and skills.

### 2.2 Episodic memory

Episodic memory is stored in a ChromaDB collection named
`episodic_memory`. Each document corresponds to a user utterance, and
metadata fields record the interaction role, a session identifier, the
user name (if known), and an importance score. When the number of
episodes exceeds a configurable threshold, DABIB calls the LLM to
summarize a batch of the least important, oldest episodes into a single
compressed document, which is re-inserted with higher importance.

This mechanism mimics coarse-grained forgetting and consolidation in
biological hippocampus-like structures while keeping storage bounded.

### 2.3 Semantic memory

Semantic memory is stored in a second ChromaDB collection named
`semantic_memory`. During a sleep phase, DABIB samples recent episodes
and a small number of older semantic facts, then asks the LLM to:

- update a narrative, third-person profile of the user;
- emit atomic personal facts (preferences, habits, biography);
- emit atomic external facts (about the world).

Personal and external facts are stored with different metadata tags and
can later be retrieved as context. This representation encourages
granular, composable knowledge instead of monolithic summaries.

### 2.4 Procedural skills

Procedural skills correspond to reusable “how-to” blocks: titles,
conditions of use and ordered lists of steps. During sleep, the LLM is
asked to extract such skills from recent episodes. Each skill is stored
as a plain text block in the `procedural_skills` collection, with
minimal metadata such as a consolidation identifier and a usage counter.

At inference time, DABIB retrieves relevant skills based on semantic
similarity to the current user message and injects them alongside episodic
and semantic context. This gives the LLM explicit access to previous
procedures instead of relying only on implicit pattern matching.

### 2.5 External fact verification

For a limited number of external facts produced in each sleep cycle,
DABIB calls a local SearxNG meta-search instance and checks whether the
fact appears supported by search results. When results are found, the
fact is marked as “supported” and associated with a small set of source
URLs and snippets; otherwise it is marked as “uncertain”.

This mechanism is intentionally cheap and conservative: it does not aim
to provide full fact-checking, but rather to flag fragile knowledge and
attach referential anchors whenever available.

---

## 3. Results and qualitative behavior

DABIB is not intended as a benchmark-winning system but as an
experimental scaffold. Qualitatively, the architecture exhibits several
desirable behaviors when integrated with a modern LLM:

- The narrative user profile becomes richer and more stable over time,
  capturing recurring preferences, biographical details and interaction
  patterns.
- Episodic memory ensures that contextual details from past sessions
  can be reintroduced even when they no longer fit in a single prompt.
- Procedural skills allow the system to reapply multi-step strategies
  (e.g. planning routines, troubleshooting checklists) in new but
  related situations.

These effects are particularly evident in long-lived sessions where the
user repeatedly interacts with the same DABIB instance.

---

## 4. Discussion

DABIB illustrates how basic ideas from cognitive neuroscience and
memory-augmented machine learning can be translated into a practical
software architecture. However, the current implementation remains a
simplified prototype:

- Memory capacity and eviction policies are coarse and mostly
  heuristic-based.
- External fact verification relies on unstructured search and does not
  model source reliability in depth.
- Skills are stored as plain text without explicit graph structure or
  parameterization.

Despite these limitations, the design highlights several research
questions, such as how to:

- weight episodic vs semantic information at inference time;
- design more explicit notion of “confidence” for external facts;
- model interference and forgetting across competing skills.

---

## 5. Limitations and future work

Key limitations of the current DABIB implementation include:

- **Evaluation** – no quantitative evaluation is provided; future work
  should compare DABIB-enabled agents with stateless baselines on
  long-horizon tasks.
- **Security and privacy** – storing rich user profiles raises privacy
  and security concerns; real deployments should incorporate encryption,
  access control and data retention policies.
- **Scalability** – the system targets single-node, research-scale
  environments; distributed storage and sharding strategies would be
  required for production use.

Planned extensions include:

- richer metadata for skills (e.g. success rates, user feedback);
- configurable, per-user memory budgets and forgetting policies;
- integration with tool-use and planning frameworks.

---

## 6. Conclusion

DABIB (Definitive Artificial Biological Imitation Brain) is a
biological-inspired memory architecture for LLMs that combines episodic
memory, semantic memory and procedural skills in a single FastAPI-based
middleware. By wrapping an existing OpenAI-compatible LLM server,
DABIB offers a practical way to experiment with long-lived, personalized
agents without modifying the underlying model.

The project is released as an open, hackable codebase intended for
researchers and practitioners interested in memory-augmented language
systems and lifelike artificial agents.
