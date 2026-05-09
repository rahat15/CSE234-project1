# Quick Reference: Exact Files to Submit

## THE ANSWER: Submit This File to Gradescope

📦 **File to Submit:** `team_138.zip` 
📍 **Location:** `/home/rabhatia/team_138.zip`
📊 **Size:** 378.8 KB

---

## What's Inside team_138.zip

```
main.py                              ← Autograder entry point (FIXED with all 5 CLI flags)
output.json                          ← Your predictions (45 questions answered)
report.pdf                           ← Your project report
generated-golden-qa-pairs.json       ← Your golden QA dataset
requirements.txt                     ← Dependencies (openai, scikit-learn)
discussion_retrieval_utils.py        ← Helper module (retrieval logic)
repo.zip                             ← Nested archive with source code
  └─ data/sourcedocs/                ← Documentation corpus
```

---

## Critical Fixes Applied

| Issue | Fixed? | Details |
|-------|--------|---------|
| Missing `--corpus-dir` flag | ✅ YES | Now required in main.py |
| Missing `--apikey-txt` flag | ✅ YES | Now required in main.py |
| Missing `--generation-model` flag | ✅ YES | Now required in main.py |
| Hardcoded API key handling | ✅ YES | Now parameterized |
| Output schema validation | ✅ YES | All 45 records valid |
| Python syntax errors | ✅ YES | Verified with ast.parse() |

---

## Download & Submit

1. **Download:** `/home/rabhatia/team_138.zip`
2. **Go to:** Canvas → [Course] → [Assignment Submission]
3. **Upload:** team_138.zip
4. **Wait:** Autograder runs ~5-10 minutes
5. **Check:** Gradescope dashboard for results

---

## Don't Forget

- ✅ API key file (`~/api-key.txt`) must exist on autograder host
- ✅ RapidFire dispatcher must be running  
- ✅ NO secrets in the archive
- ✅ NO environment variables needed

That's it! You're ready to submit. 🎉
