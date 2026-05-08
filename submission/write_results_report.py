import json
from pathlib import Path


summary_path = Path("rag_outputs/output_char_128_96_reference_like_1024.summary.json")
judge_path = Path("rag_outputs/output_char_128_96_reference_like_1024.judge.json")
output_path = Path("rag_outputs/output_char_128_96_reference_like_1024.json")
all_config_path = Path("all_config_results.json")
rapidfire_log = Path("rapidfireai/logs/discussion1-rag-docs-tuning-8configs/rapidfire.log")
report_path = Path("rag_results_summary.txt")

summary = json.loads(summary_path.read_text())
judge = json.loads(judge_path.read_text())
outputs = json.loads(output_path.read_text())
all_config_results = json.loads(all_config_path.read_text())

config_labels = {
    "1": "word_192_16, api-gpt-oss-120b, max_completion_tokens=1024",
    "2": "word_128_32, api-gpt-oss-120b, max_completion_tokens=1024",
    "3": "char_128_96, api-gpt-oss-120b, max_completion_tokens=1024",
    "4": "word_192_16, api-gpt-oss-120b, max_completion_tokens=1536",
    "5": "word_96_64, api-gpt-oss-120b, max_completion_tokens=1024",
    "6": "word_96_32, api-gpt-oss-120b, max_completion_tokens=1024",
    "7": "char_96_64, api-gpt-oss-120b, max_completion_tokens=1024",
    "8": "word_192_64, api-gpt-oss-120b, max_completion_tokens=1024",
}

lines = []
lines.append("RapidFire Documentation RAG Results")
lines.append("=" * 40)
lines.append("")
lines.append("Best configuration")
lines.append("-" * 18)
lines.append("Notebook: discussion1-rag-scifact.ipynb")
lines.append("Data folder: data/")
lines.append("Retriever: custom TF-IDF char_wb ngrams")
lines.append("Retrieval config: char_128_96")
lines.append("Generator model: api-gpt-oss-120b")
lines.append("Max completion tokens: 1024")
lines.append("Triton base URL: https://tritonai-api.ucsd.edu/v1")
lines.append("")
lines.append("Score summary")
lines.append("-" * 13)
lines.append(f"Precision@5: {summary['precision@5']:.6f}")
lines.append(f"Recall@5: {summary['recall@5']:.6f}")
lines.append(f"F1@5: {summary['f1@5']:.6f}")
lines.append(f"Retrieval Score: {summary['retrieval_score']:.6f}")
lines.append(f"Partial Generation Score: {summary['partial_generation_score']:.6f}")
lines.append(f"Partial Total Score: {summary['partial_total_score']:.6f}")
lines.append("")
lines.append("All RapidFire config results")
lines.append("-" * 28)
lines.append(
    "Run | Config | Retrieval Score | Generation Overlap Score | Total Score | MRR | Throughput | Status"
)
lines.append("-" * 120)
for result in sorted(all_config_results, key=lambda row: int(row["run_name"])):
    run_name = str(result["run_name"])
    metrics = result["metrics"]
    lines.append(
        f"{run_name} | {config_labels.get(run_name, 'unknown')} | "
        f"{metrics.get('Retrieval Score', 0):.6f} | "
        f"{metrics.get('Generation Overlap Score', 0):.6f} | "
        f"{metrics.get('Total Score', 0):.6f} | "
        f"{metrics.get('MRR', 0):.6f} | "
        f"{metrics.get('Throughput', 0):.6f} | "
        f"{result['status']}"
    )
lines.append("")
lines.append("Important note on all-config table")
lines.append("-" * 34)
lines.append(
    "The all-config RapidFire table uses the notebook's fast Generation Overlap Score, "
    "which is a local lexical proxy. The official-style LLM judge was run for the best "
    "config only, because judging every config is much slower and costs additional API calls."
)
lines.append("")
lines.append("Judge summary")
lines.append("-" * 13)
for key, value in summary["judge_summary"].items():
    if isinstance(value, (int, float)):
        lines.append(f"{key}: {value:.6f}")
    else:
        lines.append(f"{key}: {value}")
lines.append("")
lines.append("Generated artifacts")
lines.append("-" * 19)
lines.append(f"Output JSON: {output_path}")
lines.append(f"Judge report JSON: {judge_path}")
lines.append(f"Summary JSON: {summary_path}")
lines.append(f"All config JSON: {all_config_path}")
lines.append(f"RapidFire log: {rapidfire_log}")
lines.append("")
lines.append("Judge metadata")
lines.append("-" * 14)
for key, value in judge.get("meta", {}).items():
    lines.append(f"{key}: {value}")
lines.append("")
lines.append("Per-question judge results")
lines.append("-" * 26)
by_qid = {int(row["question_id"]): row for row in outputs}
for row in judge.get("per_question", []):
    qid = int(row["question_id"])
    cfg = by_qid.get(qid, {}).get("config", {})
    sources = by_qid.get(qid, {}).get("retrieved_source_files", [])
    lines.append(
        "Q{qid}: correctness={correctness}, faithfulness={faithfulness}, "
        "completeness={completeness}, status={status}, sources={sources}, config={config}".format(
            qid=qid,
            correctness=row.get("correctness"),
            faithfulness=row.get("faithfulness"),
            completeness=row.get("completeness"),
            status=row.get("status"),
            sources=", ".join(sources),
            config=cfg,
        )
    )

lines.append("")
lines.append("First five generated answers")
lines.append("-" * 28)
for row in outputs[:5]:
    lines.append(f"Q{row['question_id']}: {row['question']}")
    lines.append(f"Retrieved sources: {', '.join(row.get('retrieved_source_files', []))}")
    lines.append("Answer:")
    lines.append(row.get("answer", "").strip())
    lines.append("")

report_path.write_text("\n".join(lines), encoding="utf-8")
print(report_path.resolve())
