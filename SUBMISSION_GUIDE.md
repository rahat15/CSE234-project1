# Autograder Submission Guide - Team 138

## ✓ Submission Status: READY

Your submission archive `team_138.zip` is ready at: `/home/rabhatia/team_138.zip` (378.8 KB)

---

## Archive Contents

```
team_138.zip
├── main.py                              (4.7 KB)
├── output.json                          (350 KB - predictions)
├── report.pdf                           (331.7 KB)
├── generated-golden-qa-pairs.json       (18.3 KB - golden QA dataset)
├── requirements.txt                     (20 bytes)
├── discussion_retrieval_utils.py        (13 KB - helper module)
└── repo.zip                             (87.3 KB - nested repo snapshot)
    └── data/sourcedocs/                 (documentation corpus)
        ├── configs.rst
        ├── difference.rst
        ├── glossary.rst
        └── ... [40+ .rst files]
```

---

## ✓ Validation Checklist

### Files Present
- ✓ `main.py` exists and is syntactically valid
- ✓ `output.json` exists and is valid JSON
- ✓ `report.pdf` exists
- ✓ `generated-golden-qa-pairs.json` exists
- ✓ `requirements.txt` exists
- ✓ Helper module `discussion_retrieval_utils.py` exists
- ✓ `data/sourcedocs/` corpus exists in repo.zip

### CLI Compliance (main.py)
- ✓ Accepts `--input` (required)
- ✓ Accepts `--output` (required)
- ✓ Accepts `--corpus-dir` (required)
- ✓ Accepts `--apikey-txt` (required)
- ✓ Accepts `--generation-model` (required)
- ✓ Uses argparse for all flags
- ✓ No hardcoded API keys (removed)
- ✓ No environment variables required

### Output Schema (output.json)
- ✓ Valid JSON format
- ✓ 45 total records
- ✓ Each record has: `question_id` (int), `answer` (str), `retrieved_context` (str), `sources` (list)
- ✓ All records have non-empty fields
- ✓ Sources properly formatted: `{"file": "...", "lines": [start, end]}`

### Dependencies (requirements.txt)
- ✓ Minimal list: `openai`, `scikit-learn`
- ✓ No conflicting version pins
- ✓ No langchain-anthropic or langchain-google-genai

### Archive Structure
- ✓ Single outer archive: `team_138.zip`
- ✓ One nested archive: `repo.zip`
- ✓ POSIX paths (forward slashes)
- ✓ No .tar.gz files
- ✓ Correct file extensions

---

## How to Submit

1. **Download the archive:**
   - File: `team_138.zip` (378.8 KB)
   - Location: `/home/rabhatia/team_138.zip`

2. **Upload to Gradescope:**
   - Go to Canvas → Course → Gradescope submission link
   - Select the assignment "Project Submission"
   - Upload `team_138.zip`
   - Autograder will automatically extract and validate

3. **Expected Timeline:**
   - Extraction: ~30 seconds
   - Autograder run: ~5-10 minutes
   - Results: Available in Gradescope submission dashboard

---

## Key Changes Made

### 1. Fixed main.py CLI (CRITICAL)
**Before:**
```bash
python main.py --input FILE --output FILE
```

**After (NOW COMPLIANT):**
```bash
python main.py \
  --input <path> \
  --output <path> \
  --corpus-dir <path> \
  --apikey-txt <path> \
  --generation-model <model_name>
```

### 2. Removed Hardcoded API Key Handling
- ✓ Removed environment variable fallback checks
- ✓ Now requires `--apikey-txt` parameter for API key
- ✓ Load function: `load_api_key(apikey_path: str | None)`

### 3. Parameterized Generation Model
- ✓ Removed hardcoded `GENERATOR_MODEL` from constant
- ✓ Now accepts `--generation-model` flag
- ✓ Passes model name to `answer_questions()` function

---

## Testing the Submission Locally

Before submitting, verify locally:

```bash
# Extract and test
unzip -q team_138.zip -d /tmp/test_submission
cd /tmp/test_submission

# Test with sample input
python main.py \
  --input generated-golden-qa-pairs.json \
  --output test_output.json \
  --corpus-dir repo_content/data/sourcedocs \
  --apikey-txt /home/rabhatia/api-key.txt \
  --generation-model "api-gpt-oss-120b"

# Verify output
python3 -c "import json; d=json.load(open('test_output.json')); print(f'✓ Generated {len(d)} predictions')"
```

---

## Troubleshooting

### If autograder fails with "main.py not found"
- ✓ Confirmed: main.py is at root of team_138.zip

### If autograder fails with "missing CLI arguments"
- ✓ Confirmed: All 5 required flags are implemented

### If autograder fails with "invalid output schema"
- ✓ Confirmed: All records have correct fields with correct types

### If autograder fails to load helper modules
- ✓ Confirmed: discussion_retrieval_utils.py is at root of archive
- ✓ Confirmed: repo.zip contains any additional sources needed

---

## File Inventory

| File | Size | Purpose | Status |
|------|------|---------|--------|
| main.py | 4.7 KB | Autograder entry point | ✓ Fixed |
| output.json | 350 KB | Predictions on released set | ✓ Valid |
| report.pdf | 331.7 KB | Project report | ✓ Included |
| generated-golden-qa-pairs.json | 18.3 KB | Golden QA dataset | ✓ Included |
| requirements.txt | 20 B | Dependencies | ✓ Minimal |
| discussion_retrieval_utils.py | 13 KB | Retrieval helper | ✓ Included |
| repo.zip | 87.3 KB | Source snapshot | ✓ Nested |

---

## Submission Metadata

- **Team ID:** 138
- **Archive Name:** team_138.zip
- **Archive Size:** 378.8 KB
- **Files Included:** 7 top-level items
- **Nested Archives:** 1 (repo.zip)
- **Created:** 2026-05-08

---

## Important Reminders

1. **Do NOT modify** `team_138.zip` after download
2. **Do NOT add secrets** to repo.zip (API keys, credentials, etc.)
3. **Ensure API key file** (`api-key.txt`) is available on autograder host
4. **RapidFire dispatcher** must be running when autograder invokes main.py
5. **CPU-only environment** - no GPU available on autograder
6. **No environment variables needed** - all config via CLI args
7. **Single config only** - no multi-sweep experiments in main.py

---

## Questions or Issues?

If the autograder rejects your submission:
1. Check that all 5 CLI flags are accepted
2. Verify output.json has all 4 required fields per record
3. Ensure requirements.txt has no version conflicts
4. Check that secrets are NOT in the archive
5. Re-run: `python -c "import ast; ast.parse(open('main.py').read())"`

Good luck with your submission! 🚀
