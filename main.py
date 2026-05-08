#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from discussion_retrieval_utils import INSTRUCTIONS, retrieve_contexts_with_sources


TRITON_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
GENERATOR_MODEL = "api-gpt-oss-120b"
RETRIEVAL_CONFIG = "char_128_96"
MAX_COMPLETION_TOKENS = 1024
MAX_CONTEXT_TOKENS = 2000


def load_api_key() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    for path in [Path("~/api.txt").expanduser(), Path("~/api-key.txt").expanduser()]:
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()[0].strip()
    raise FileNotFoundError("Set OPENAI_API_KEY or provide ~/api.txt / ~/api-key.txt")


def estimate_tokens(text: str) -> int:
    # Conservative enough for the project budget without adding a tokenizer dependency.
    return max(1, len(text.split()))


def fit_context_to_budget(context_chunks: list[str], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    kept = []
    total = 0
    for chunk in context_chunks:
        chunk_tokens = estimate_tokens(chunk)
        if kept and total + chunk_tokens > max_tokens:
            break
        if not kept and chunk_tokens > max_tokens:
            words = chunk.split()
            kept.append(" ".join(words[:max_tokens]))
            break
        kept.append(chunk)
        total += chunk_tokens
    return "\n\n---\n\n".join(kept)


def answer_questions(input_rows: list[dict], client: OpenAI) -> list[dict]:
    questions = [row["question"] for row in input_rows]
    contexts, _source_files, source_details = retrieve_contexts_with_sources(
        questions, RETRIEVAL_CONFIG
    )

    output_rows = []
    for row, context_chunks, sources in zip(input_rows, contexts, source_details):
        retrieved_context = fit_context_to_budget(context_chunks)
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{row['question']}\n\n"
                        f"RETRIEVED CONTEXT:\n{retrieved_context}\n\n"
                        "ANSWER:"
                    ),
                },
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
        output_rows.append(
            {
                "question_id": int(row["question_id"]),
                "answer": answer,
                "retrieved_context": retrieved_context,
                "sources": sources,
            }
        )
    return output_rows


def load_input_rows(input_path: Path) -> list[dict]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of question objects.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Input row {index} must be an object.")
        if "question_id" not in row or "question" not in row:
            raise ValueError(f"Input row {index} must include question_id and question.")
        if not isinstance(row["question"], str):
            raise ValueError(f"Input row {index} has a non-string question.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final RapidFire documentation RAG pipeline.")
    parser.add_argument("--input", required=True, help="Input JSON list with question_id and question.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    input_rows = load_input_rows(Path(args.input))
    client = OpenAI(api_key=load_api_key(), base_url=TRITON_BASE_URL)
    output_rows = answer_questions(input_rows, client)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
