import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

from discussion_retrieval_utils import RETRIEVAL_CONFIGS, retrieve_contexts


DATA_ROOT = Path("data")
VALIDATION_PATH = DATA_ROOT / "validation" / "validation-set-golden-qa-pairs.json"
TRITON_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
KEY_PATH = next(path for path in [Path("~/api.txt").expanduser(), Path("~/api-key.txt").expanduser()] if path.exists())
API_KEY = KEY_PATH.read_text(encoding="utf-8").splitlines()[0].strip()


PROMPTS = {
    "concise": """You answer questions about RapidFire AI documentation.
Use only the retrieved context. Answer with the exact API names, parameters, and distinctions needed.
Be concise but complete. Cite source file names when helpful.
If the context does not contain the answer, say what is missing.""",
    "reference_like": """You answer questions about RapidFire AI documentation.
Use only the retrieved context. Write a direct answer that covers every substantive part of the question.
Prefer precise documentation wording and include key parameter names, class names, defaults, and constraints.
Do not add unsupported claims. Cite source file names briefly when helpful.""",
}


def generate_output(config_name: str, prompt_name: str, model: str, max_tokens: int, out_path: Path):
    client = OpenAI(api_key=API_KEY, base_url=TRITON_BASE_URL)
    examples = json.loads(VALIDATION_PATH.read_text())
    questions = [ex["question"] for ex in examples]
    contexts, sources = retrieve_contexts(questions, config_name)
    rows = []
    for idx, (ex, context_chunks, source_files) in enumerate(zip(examples, contexts, sources), 1):
        retrieved_context = "\n\n---\n\n".join(context_chunks)
        messages = [
            {"role": "system", "content": PROMPTS[prompt_name]},
            {
                "role": "user",
                "content": f"QUESTION:\n{ex['question']}\n\nRETRIEVED CONTEXT:\n{retrieved_context}\n\nANSWER:",
            },
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
        rows.append(
            {
                "question_id": int(ex["question_id"]),
                "question": ex["question"],
                "answer": answer,
                "retrieved_context": retrieved_context,
                "retrieved_source_files": source_files,
                "config": {
                    "retrieval_config": config_name,
                    "prompt": prompt_name,
                    "model": model,
                    "max_completion_tokens": max_tokens,
                },
            }
        )
        print(f"[{config_name}/{prompt_name}] generated {idx}/{len(examples)}", flush=True)
    out_path.write_text(json.dumps(rows, indent=2))


def run_judge(output_path: Path, judge_model: str):
    cmd = [
        sys.executable,
        "Metrics/run_judge.py",
        "--output",
        str(output_path),
        "--validation",
        str(VALIDATION_PATH),
        "--base-url",
        TRITON_BASE_URL,
        "--model",
        judge_model,
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = API_KEY
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    report_path = output_path.with_suffix(".judge.json")
    report_path.write_text(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    report = json.loads(proc.stdout)
    return report, report_path


def retrieval_score_for_output(output_path: Path):
    examples = {int(ex["question_id"]): ex for ex in json.loads(VALIDATION_PATH.read_text())}
    rows = json.loads(output_path.read_text())
    precisions, recalls, f1s = [], [], []
    for row in rows:
        expected = {ev["file"] for ev in examples[int(row["question_id"])]["source_evidence"]}
        top = list(dict.fromkeys(row["retrieved_source_files"]))[:5]
        hits = len(set(top) & expected)
        p = hits / len(top) if top else 0.0
        r = hits / len(expected) if expected else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = sum(f1s) / len(f1s)
    return {
        "precision@5": precision,
        "recall@5": recall,
        "f1@5": f1,
        "retrieval_score": (precision + recall + f1) / 3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="char_128_96", choices=sorted(RETRIEVAL_CONFIGS))
    parser.add_argument("--prompt", default="reference_like", choices=sorted(PROMPTS))
    parser.add_argument("--model", default="api-gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    out_dir = Path("rag_outputs")
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / f"output_{args.config}_{args.prompt}_{args.max_tokens}.json"
    generate_output(args.config, args.prompt, args.model, args.max_tokens, output_path)
    judge_report, report_path = run_judge(output_path, args.judge_model)
    retrieval = retrieval_score_for_output(output_path)
    gen = judge_report["summary"]["Partial Generation Score"]
    total = 0.5 * retrieval["retrieval_score"] + 0.5 * gen
    summary = {
        "output": str(output_path),
        "judge_report": str(report_path),
        **retrieval,
        "partial_generation_score": gen,
        "partial_total_score": total,
        "judge_summary": judge_report["summary"],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
