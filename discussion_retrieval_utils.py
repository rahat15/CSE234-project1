import math
import re
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODULE_DIR = Path(__file__).resolve().parent
DATA_ROOT_CANDIDATES = [
    Path(os.environ["RAG_DATA_DIR"]).expanduser()
    for _ in [0]
    if os.environ.get("RAG_DATA_DIR")
] + [
    MODULE_DIR / "data",
    Path.cwd() / "data",
    Path.home() / "data",
]
DATA_ROOT = next((path for path in DATA_ROOT_CANDIDATES if path.exists()), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find the RAG data directory. Put the corpus in ./data or set RAG_DATA_DIR."
    )
SOURCE_DOCS_DIR = DATA_ROOT / "sourcedocs"
if not SOURCE_DOCS_DIR.exists():
    raise FileNotFoundError(f"Missing source documents directory: {SOURCE_DOCS_DIR}")

INSTRUCTIONS = """
You answer questions about the RapidFire AI OSS documentation.
Use only the provided retrieved evidence. Be complete, faithful to the evidence, and concise.
When the evidence supports the answer, cite the relevant source file names in prose.
If the evidence is insufficient, say what is missing rather than inventing details.
"""

RETRIEVAL_CONFIGS = {
    "word_192_16": {
        "mode": "chunk",
        "chunk_size": 192,
        "chunk_overlap": 16,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 2,
        "top_chunks": 25,
        "context_chunks": 6,
        "top_k_files": 5,
    },
    "word_128_32": {
        "mode": "chunk",
        "chunk_size": 128,
        "chunk_overlap": 32,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 2,
        "top_chunks": 25,
        "context_chunks": 7,
        "top_k_files": 5,
    },
    "char_128_96": {
        "mode": "chunk",
        "chunk_size": 128,
        "chunk_overlap": 96,
        "analyzer": "char_wb",
        "ngram_min": 3,
        "ngram_max": 5,
        "top_chunks": 25,
        "context_chunks": 7,
        "top_k_files": 5,
    },
    "word_96_64": {
        "mode": "chunk",
        "chunk_size": 96,
        "chunk_overlap": 64,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 2,
        "top_chunks": 25,
        "context_chunks": 7,
        "top_k_files": 5,
    },
    "word_96_32": {
        "mode": "chunk",
        "chunk_size": 96,
        "chunk_overlap": 32,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 3,
        "top_chunks": 25,
        "context_chunks": 7,
        "top_k_files": 5,
    },
    "char_96_64": {
        "mode": "chunk",
        "chunk_size": 96,
        "chunk_overlap": 64,
        "analyzer": "char_wb",
        "ngram_min": 3,
        "ngram_max": 5,
        "top_chunks": 25,
        "context_chunks": 7,
        "top_k_files": 5,
    },
    "word_192_64": {
        "mode": "chunk",
        "chunk_size": 192,
        "chunk_overlap": 64,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 2,
        "top_chunks": 25,
        "context_chunks": 6,
        "top_k_files": 5,
    },
}


def _split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def _line_span_for_text(full_text: str, part: str) -> list[int]:
    start_char = full_text.find(part)
    if start_char < 0:
        return [1, max(1, full_text.count("\n") + 1)]
    end_char = start_char + len(part)
    start_line = full_text.count("\n", 0, start_char) + 1
    end_line = full_text.count("\n", 0, end_char) + 1
    return [start_line, max(start_line, end_line)]


def _load_doc_chunks(chunk_size: int, overlap: int, mode: str) -> list[dict[str, Any]]:
    chunks = []
    for path in sorted(SOURCE_DOCS_DIR.glob("*.rst")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts = [text] if mode == "file" else _split_words(text, chunk_size, overlap)
        for idx, part in enumerate(parts):
            if part.strip():
                chunks.append(
                    {
                        "source_file": path.name,
                        "chunk_id": idx,
                        "text": part,
                        "lines": _line_span_for_text(text, part),
                    }
                )
    return chunks


@lru_cache(maxsize=None)
def _build_tfidf_retriever(
    chunk_size: int,
    overlap: int,
    mode: str,
    analyzer: str,
    ngram_min: int,
    ngram_max: int,
):
    chunks = _load_doc_chunks(chunk_size, overlap, mode)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english" if analyzer == "word" else None,
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        sublinear_tf=True,
        max_df=0.95,
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks, vectorizer, matrix


def retrieve_contexts(
    queries: list[str], retrieval_config_name: str
) -> tuple[list[list[str]], list[list[str]]]:
    contexts, sources, _source_details = retrieve_contexts_with_sources(queries, retrieval_config_name)
    return contexts, sources


def retrieve_contexts_with_sources(
    queries: list[str], retrieval_config_name: str
) -> tuple[list[list[str]], list[list[str]], list[list[dict[str, Any]]]]:
    retrieval_config = RETRIEVAL_CONFIGS[retrieval_config_name]
    chunks, vectorizer, matrix = _build_tfidf_retriever(
        retrieval_config["chunk_size"],
        retrieval_config["chunk_overlap"],
        retrieval_config["mode"],
        retrieval_config["analyzer"],
        retrieval_config["ngram_min"],
        retrieval_config["ngram_max"],
    )
    query_matrix = vectorizer.transform(queries)
    similarities = cosine_similarity(query_matrix, matrix)
    all_contexts, all_sources, all_source_details = [], [], []
    for row in similarities:
        top_indices = row.argsort()[::-1][: retrieval_config["top_chunks"]]
        selected_chunks = [chunks[i] for i in top_indices]

        contexts = []
        sources = []
        source_details = []
        seen_files = set()
        for chunk in selected_chunks:
            source_file = chunk["source_file"]
            sources.append(source_file)
            source_details.append({"file": source_file, "lines": chunk["lines"]})
            if source_file not in seen_files or len(contexts) < retrieval_config["context_chunks"]:
                contexts.append(
                    f"Source file: {source_file}\n"
                    f"Chunk: {chunk['chunk_id']}\n"
                    f"{chunk['text']}"
                )
                seen_files.add(source_file)
            if len(contexts) >= retrieval_config["context_chunks"]:
                break

        all_contexts.append(contexts)
        all_sources.append(list(dict.fromkeys(sources))[: retrieval_config["top_k_files"]])
        unique_source_details = []
        seen_details = set()
        for detail in source_details:
            key = (detail["file"], tuple(detail["lines"]))
            if key not in seen_details:
                unique_source_details.append(detail)
                seen_details.add(key)
            if len(unique_source_details) >= retrieval_config["top_k_files"]:
                break
        all_source_details.append(unique_source_details)
    return all_contexts, all_sources, all_source_details


def _preprocess_with_config(batch, retrieval_config_name: str):
    contexts, retrieved_documents = retrieve_contexts(batch["query"], retrieval_config_name)
    serialized_contexts = ["\n\n---\n\n".join(context) for context in contexts]
    return {
        "prompts": [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Retrieved evidence:\n{context}\n\n"
                        "Answer:"
                    ),
                },
            ]
            for question, context in zip(batch["query"], serialized_contexts)
        ],
        "retrieved_documents": retrieved_documents,
        "retrieval_config_name": [retrieval_config_name] * len(batch["query"]),
        **batch,
    }


def preprocess_word_192_16(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "word_192_16")


def preprocess_word_128_32(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "word_128_32")


def preprocess_char_128_96(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "char_128_96")


def preprocess_word_96_64(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "word_96_64")


def preprocess_word_96_32(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "word_96_32")


def preprocess_char_96_64(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "char_96_64")


def preprocess_word_192_64(batch, rag, prompt_manager):
    return _preprocess_with_config(batch, "word_192_64")


def sample_postprocess_fn(batch):
    batch["ground_truth_documents"] = batch["expected_source_files"]
    batch["answer"] = batch["generated_text"]
    return batch


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def compute_retrieval_metrics(retrieved_docs: list[str], expected_docs: set[str], k=5):
    top = list(dict.fromkeys(retrieved_docs))[:k]
    hits = len(set(top).intersection(expected_docs))
    precision = hits / len(top) if top else 0.0
    recall = hits / len(expected_docs) if expected_docs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    relevance = [1 if doc in expected_docs else 0 for doc in top]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = [1] * min(k, len(expected_docs)) + [0] * max(0, k - len(expected_docs))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0
    rr = next((1 / (i + 1) for i, doc in enumerate(top) if doc in expected_docs), 0.0)
    hit_rate = 1.0 if hits else 0.0
    return precision, recall, f1, ndcg, rr, hit_rate


def generation_overlap_score(answer: str, reference_answer: str) -> float:
    answer_tokens = _token_set(answer)
    reference_tokens = _token_set(reference_answer)
    if not answer_tokens or not reference_tokens:
        return 0.0
    precision = len(answer_tokens & reference_tokens) / len(answer_tokens)
    recall = len(answer_tokens & reference_tokens) / len(reference_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return min(1.0, f1 * 1.35)


def sample_compute_metrics_fn(batch):
    precisions, recalls, f1_scores, ndcgs, rrs, hit_rates, gen_scores = [], [], [], [], [], [], []
    total_queries = len(batch["query"])
    for pred, gt, answer, reference in zip(
        batch["retrieved_documents"],
        batch["ground_truth_documents"],
        batch["answer"],
        batch["reference_answer"],
    ):
        precision, recall, f1, ndcg, rr, hit_rate = compute_retrieval_metrics(pred, set(gt), k=5)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ndcgs.append(ndcg)
        rrs.append(rr)
        hit_rates.append(hit_rate)
        gen_scores.append(generation_overlap_score(answer, reference))

    precision = sum(precisions) / total_queries
    recall = sum(recalls) / total_queries
    f1 = sum(f1_scores) / total_queries
    retrieval_score = (precision + recall + f1) / 3
    generation_score = sum(gen_scores) / total_queries
    total_score = 0.5 * retrieval_score + 0.5 * generation_score

    return {
        "Total": {"value": total_queries},
        "Precision@5": {"value": precision},
        "Recall@5": {"value": recall},
        "F1@5": {"value": f1},
        "NDCG@5": {"value": sum(ndcgs) / total_queries},
        "MRR": {"value": sum(rrs) / total_queries},
        "Hit Rate@5": {"value": sum(hit_rates) / total_queries},
        "Retrieval Score": {"value": retrieval_score},
        "Generation Overlap Score": {"value": generation_score},
        "Total Score": {"value": total_score},
    }


def sample_accumulate_metrics_fn(aggregated_metrics):
    num_queries_per_batch = [m["value"] for m in aggregated_metrics["Total"]]
    total_queries = sum(num_queries_per_batch)
    algebraic_metrics = [
        "Precision@5",
        "Recall@5",
        "F1@5",
        "NDCG@5",
        "MRR",
        "Hit Rate@5",
        "Retrieval Score",
        "Generation Overlap Score",
        "Total Score",
    ]
    return {
        "Total": {"value": total_queries},
        **{
            metric: {
                "value": sum(
                    m["value"] * queries
                    for m, queries in zip(aggregated_metrics[metric], num_queries_per_batch)
                )
                / total_queries,
                "is_algebraic": True,
                "value_range": (0, 1),
            }
            for metric in algebraic_metrics
        },
    }
