# RapidFire Documentation RAG

This repository contains a deterministic RAG pipeline for answering questions over the RapidFire AI OSS documentation snapshot in `data/sourcedocs`.

## Run

Set the TritonAI key using either `OPENAI_API_KEY`, `~/api.txt`, or `~/api-key.txt`, then run:

```bash
python3 main.py --input validation_input.json --output output.json
```

The code looks for the corpus in `data/sourcedocs` next to the repository. If the data is stored elsewhere, set `RAG_DATA_DIR` to the directory that contains `sourcedocs`.

The input JSON must be a list of objects with:

- `question_id`
- `question`

The output JSON contains:

- `question_id`
- `answer`
- `retrieved_context`
- `sources`

## Final Pipeline

- Retriever: custom TF-IDF over `.rst` documentation chunks
- Final retrieval config: `char_128_96`
- Generator: TritonAI OpenAI-compatible API
- Model: `api-gpt-oss-120b`
- Max completion tokens: `1024`
- Context budget: capped below 2,000 estimated tokens per query

## Results

The best judged validation output is summarized in:

- `rag_results_summary.txt`
- `rag_outputs/output_char_128_96_reference_like_1024.summary.json`
- `rag_outputs/output_char_128_96_reference_like_1024.judge.json`

The final runnable pipeline output is:

- `output.json`

RapidFire logs are copied into:

- `logs/rapidfire-8configs.log`

## Dependencies

Python dependencies used by the final pipeline:

- `openai`
- `scikit-learn`

The notebook experiments additionally use RapidFire AI and its notebook dependencies.
