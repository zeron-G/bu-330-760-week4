# Week 4 Starter: Math Agent

A ReAct agent that solves questions using tool calls.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it.

2. Copy `.env.example` to `.env` and choose a backend:
   ```bash
   cp .env.example .env
   ```
   The app now supports three modes:
   - `MODEL_PROVIDER=auto`: prefer Codex OAuth from `~/.codex/auth.json`, then OpenAI API, then Google AI Studio
   - `MODEL_PROVIDER=codex`: force the migrated Codex OAuth backend
   - `MODEL_PROVIDER=openai` or `google`: force a specific API provider

   Codex OAuth uses the migrated ChatGPT/Codex flow and does not require copying a key into this repo.

3. Make sure `.env` is in your `.gitignore` so you don't commit your key.

## Run

```bash
uv run agent.py
```

uv will install dependencies automatically on first run.

The agent will work through each question in `math_questions.md` and print the ReAct trace (Reason / Act / Result) for each one.

## What Changed

- Implemented `product_lookup` so the agent can price items from `products.json`
- Added automatic backend selection with a migrated Codex OAuth option
- Kept OpenAI API key fallback support for direct API usage

## Video

Video link: https://youtu.be/a7ep2wgSHXg

## Files

- `agent.py` - the ReAct agent (this is the file you'll modify)
- `calculator.py` - calculator tool
- `products.json` - product catalog with prices
- `math_questions.md` - the questions the agent solves
- `.env.example` - template for your API key
