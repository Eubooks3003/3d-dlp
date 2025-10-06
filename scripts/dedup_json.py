# dedupe_index.py
import json, sys
p = sys.argv[1]
j = json.load(open(p))
seen = set()
out = []
dups = 0
for it in j["items"]:
    key = (it.get("cam"), it.get("ep"), int(it.get("t")))
    if key in seen:
        dups += 1
        continue
    seen.add(key)
    out.append(it)
j["items"] = out
json.dump(j, open(p, "w"), indent=2)
print(f"kept {len(out)} items; removed {dups} duplicates")
