import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_ROOT = Path("data")
DOC_DIR = DATA_ROOT / "sourcedocs"
QUESTIONS = json.loads((DATA_ROOT / "validation" / "validation-set-golden-qa-pairs.json").read_text())


@dataclass
class Chunk:
    file: str
    text: str


def tokenize(text):
    return re.findall(r"[A-Za-z0-9_./:-]+", text.lower())


def split_words(text, chunk_size, overlap):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def make_chunks(chunk_size, overlap, mode):
    chunks = []
    for path in sorted(DOC_DIR.glob("*.rst")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if mode == "file":
            parts = [text]
        else:
            parts = split_words(text, chunk_size, overlap)
        chunks.extend(Chunk(path.name, part) for part in parts if part.strip())
    return chunks


def ndcg(retrieved, expected, k=5):
    gains = [1 if doc in expected else 0 for doc in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = [1] * min(k, len(expected)) + [0] * max(0, k - len(expected))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def score_rankings(rankings):
    ps, rs, fs, hs, ns, ms = [], [], [], [], [], []
    for ex, retrieved in zip(QUESTIONS, rankings):
        expected = {ev["file"] for ev in ex["source_evidence"]}
        top = list(dict.fromkeys(retrieved))[:5]
        hit = len(set(top) & expected)
        p = hit / len(top) if top else 0
        r = hit / len(expected) if expected else 0
        f = 2 * p * r / (p + r) if p + r else 0
        rr = next((1 / (i + 1) for i, doc in enumerate(top) if doc in expected), 0)
        ps.append(p)
        rs.append(r)
        fs.append(f)
        hs.append(1.0 if hit else 0.0)
        ns.append(ndcg(top, expected))
        ms.append(rr)
    retrieval = (sum(ps) / len(ps) + sum(rs) / len(rs) + sum(fs) / len(fs)) / 3
    return {
        "score": retrieval,
        "precision@5": sum(ps) / len(ps),
        "recall@5": sum(rs) / len(rs),
        "f1@5": sum(fs) / len(fs),
        "hit@5": sum(hs) / len(hs),
        "ndcg@5": sum(ns) / len(ns),
        "mrr": sum(ms) / len(ms),
    }


def run_config(mode, chunk_size, overlap, analyzer, ngram_range, use_ref_answer=False, top_chunks=25):
    chunks = make_chunks(chunk_size, overlap, mode)
    corpus = [c.text for c in chunks]
    queries = [
        ex["question"] + (("\n" + ex["reference_answer"]) if use_ref_answer else "")
        for ex in QUESTIONS
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        analyzer=analyzer,
        ngram_range=ngram_range,
        sublinear_tf=True,
        max_df=0.95,
    )
    matrix = vectorizer.fit_transform(corpus)
    q_matrix = vectorizer.transform(queries)
    sims = cosine_similarity(q_matrix, matrix)
    rankings = []
    for row in sims:
        idxs = row.argsort()[::-1][:top_chunks]
        rankings.append([chunks[i].file for i in idxs])
    metrics = score_rankings(rankings)
    return metrics, len(chunks)


configs = []
for mode in ["file", "chunk"]:
    sizes = [99999] if mode == "file" else [96, 128, 192, 256, 384, 512, 768]
    for size in sizes:
        for overlap in ([0] if mode == "file" else [16, 32, 64, 96]):
            if overlap >= size:
                continue
            for analyzer, ngrams in [("word", (1, 2)), ("word", (1, 3)), ("char_wb", (3, 5)), ("char_wb", (4, 6))]:
                configs.append((mode, size, overlap, analyzer, ngrams))

results = []
for cfg in configs:
    metrics, n_chunks = run_config(*cfg)
    results.append((metrics["score"], metrics, n_chunks, cfg))

for score, metrics, n_chunks, cfg in sorted(results, key=lambda x: x[0], reverse=True)[:20]:
    print(f"{score:.4f}", cfg, "chunks", n_chunks, metrics)
