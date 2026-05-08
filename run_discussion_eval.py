import json


nb = json.load(open("discussion1-rag-scifact.ipynb"))
ns = {}
cells = [2, 3, 6, 8, 10, 12, 13, 15, 17, 19, 21, 23, 25, 27]

for i in cells:
    print(f"RUNNING CELL {i}", flush=True)
    code = "".join(nb["cells"][i]["source"])
    exec(compile(code, f"cell_{i}", "exec"), ns)

print("FINAL_RESULTS")
print(ns["results_df"].sort_values("Total Score", ascending=False).to_string(index=False))
