"""Math agent that solves questions using tools in a ReAct loop."""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from calculator import calculate
from codex_oauth import CodexOAuthManager, build_codex_model

load_dotenv()

ROOT = Path(__file__).resolve().parent
PRODUCTS_PATH = ROOT / "products.json"


def build_model():
    """Choose the best available model backend from the local environment."""
    provider = os.getenv("MODEL_PROVIDER", "auto").strip().lower()
    codex_model = os.getenv("CODEX_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    if provider in {"auto", "codex"} and CodexOAuthManager.is_available():
        return build_codex_model(codex_model)
    if provider == "codex":
        raise RuntimeError("MODEL_PROVIDER=codex, but ~/.codex/auth.json is unavailable or incomplete.")

    if provider in {"auto", "openai"} and os.getenv("OPENAI_API_KEY", "").strip():
        return OpenAIModel(openai_model)
    if provider == "openai":
        raise RuntimeError("MODEL_PROVIDER=openai, but OPENAI_API_KEY is missing.")

    if provider in {"auto", "google"} and os.getenv("GOOGLE_API_KEY", "").strip():
        return f"google-gla:{google_model}"
    if provider == "google":
        raise RuntimeError("MODEL_PROVIDER=google, but GOOGLE_API_KEY is missing.")

    raise RuntimeError(
        "No usable model backend found. Configure Codex OAuth, OPENAI_API_KEY, or GOOGLE_API_KEY."
    )


MODEL = build_model()

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, float]:
    with PRODUCTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.

    Use this when a question asks about product prices from the catalog.
    """
    catalog = load_catalog()
    normalized = product_name.strip().casefold().rstrip(".?!")
    aliases = {normalized}
    if normalized.endswith("s"):
        aliases.add(normalized[:-1])
    else:
        aliases.add(f"{normalized}s")

    for name, price in catalog.items():
        canonical = name.casefold()
        if canonical in aliases or canonical.rstrip("s") in aliases:
            return f"{price:.2f}"

    available = ", ".join(sorted(catalog))
    return f"Product not found. Available products: {available}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with (ROOT / path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = agent.run_sync(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()
