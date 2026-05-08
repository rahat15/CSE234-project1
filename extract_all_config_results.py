import json
import sqlite3
from collections import defaultdict
from pathlib import Path


DB = Path("rapidfireai/db/rapidfire_mlflow.db")
EXPERIMENT_NAME = "discussion1-rag-docs-tuning-8configs"


def rows_as_dicts(cur, query, args=()):
    cur.execute(query, args)
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


con = sqlite3.connect(DB)
cur = con.cursor()

experiments = rows_as_dicts(
    cur,
    "select experiment_id, name from experiments where name = ?",
    (EXPERIMENT_NAME,),
)
if not experiments:
    raise SystemExit(f"Experiment not found in MLflow DB: {EXPERIMENT_NAME}")

experiment_id = experiments[0]["experiment_id"]
runs = rows_as_dicts(
    cur,
    "select run_uuid, name, start_time, end_time, status from runs where experiment_id = ? order by start_time",
    (experiment_id,),
)

all_results = []
for run in runs:
    run_uuid = run["run_uuid"]
    params = {
        row["key"]: row["value"]
        for row in rows_as_dicts(cur, "select key, value from params where run_uuid = ?", (run_uuid,))
    }
    latest_metrics = {
        row["key"]: row["value"]
        for row in rows_as_dicts(
            cur,
            "select key, value from latest_metrics where run_uuid = ?",
            (run_uuid,),
        )
    }
    tags = {
        row["key"]: row["value"]
        for row in rows_as_dicts(cur, "select key, value from tags where run_uuid = ?", (run_uuid,))
    }
    all_results.append(
        {
            "run_uuid": run_uuid,
            "run_name": run["name"],
            "status": run["status"],
            "params": params,
            "metrics": latest_metrics,
            "tags": tags,
        }
    )

Path("all_config_results.json").write_text(json.dumps(all_results, indent=2))

print(f"experiment_id={experiment_id}")
print(f"runs={len(all_results)}")
for result in all_results:
    print("---")
    print(result["run_name"], result["run_uuid"], result["status"])
    interesting_params = {
        k: v
        for k, v in result["params"].items()
        if "config" in k.lower()
        or "model" in k.lower()
        or "preprocess" in k.lower()
        or "completion" in k.lower()
    }
    print("params", interesting_params)
    print("metrics", result["metrics"])
