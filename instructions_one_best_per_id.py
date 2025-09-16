# Keep ONE best instruction per _row_id (max delta)

import json, collections, statistics, os

IN_JSONL  = "/kaggle/input/dataset-instructions-filtered/instructions_ru_sbert_filtered_with_negs.jsonl"
OUT_JSONL = "/kaggle/working/instructions_one_best_per_id.jsonl"


MIN_DELTA = None  

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

rows = list(read_jsonl(IN_JSONL))
print("Loaded lines:", len(rows))

by_id = collections.defaultdict(list)
for r in rows:
    rid = r.get("_row_id","")
    if not rid: 
        continue
    by_id[rid].append(r)

kept = []
deltas = []
styles = collections.Counter()
dropped_for_min = 0
weird_inconsistency = 0

def _score(r):
    d = r.get("delta", None)
    if d is None:
        return (-1e9, r.get("sim_with_instruction", -1e9))
    return (float(d), float(r.get("sim_with_instruction", -1e9)))

for rid, cand_list in by_id.items():
    cand_list_sorted = sorted(cand_list, key=_score, reverse=True)
    best = cand_list_sorted[0]
    best_delta = best.get("delta", None)
    if MIN_DELTA is not None and (best_delta is None or float(best_delta) < float(MIN_DELTA)):
        dropped_for_min += 1
        continue

    qset = {c.get("query_ru","") for c in cand_list}
    pset = {c.get("positive_ru","") for c in cand_list}
    if len(qset) > 1 or len(pset) > 1:
        weird_inconsistency += 1

    kept.append({
        "_row_id": rid,
        "query_ru": best.get("query_ru",""),
        "positive_ru": best.get("positive_ru",""),
        "hard_negs_ru": best.get("hard_negs_ru", []),

        "query_en": best.get("query_en",""),
        "positive_en": best.get("positive_en",""),
        "hard_negs_en": best.get("hard_negs_en", []),

        "style": best.get("style",""),
        "length_format": best.get("length_format",""),
        "instruction": best.get("instruction",""),

        "base_sim": best.get("base_sim", None),
        "sim_with_instruction": best.get("sim_with_instruction", None),
        "delta": best.get("delta", None),
    })

    if best.get("delta") is not None:
        deltas.append(float(best["delta"]))
    styles[best.get("style","")] += 1

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n=== One-best filter summary ===")
print("Unique _row_id in input:", len(by_id))
print("Kept rows (one per _row_id):", len(kept))
print("Dropped due to MIN_DELTA:", dropped_for_min, "| MIN_DELTA =", MIN_DELTA)
print("Rows with inconsistent base fields (query/positive varied within same _row_id):", weird_inconsistency)
if deltas:
    print("Best-delta avg:", round(statistics.mean(deltas),4), "| median:", round(statistics.median(deltas),4),
          "| >0:", sum(d>0 for d in deltas), "| =0:", sum(d==0 for d in deltas), "| <0:", sum(d<0 for d in deltas))
print("By style:", dict(styles))
print("Saved to:", OUT_JSONL)




"""
Loaded lines: 5624

=== One-best filter summary ===
Unique _row_id in input: 2052
Kept rows (one per _row_id): 2052
Dropped due to MIN_DELTA: 0 | MIN_DELTA = None
Rows with inconsistent base fields (query/positive varied within same _row_id): 0
Best-delta avg: 0.148 | median: 0.1379 | >0: 2052 | =0: 0 | <0: 0
By style: {'persona': 643, 'short_strict': 779, 'background_long': 630}
Saved to: /kaggle/working/instructions_one_best_per_id.jsonl
"""