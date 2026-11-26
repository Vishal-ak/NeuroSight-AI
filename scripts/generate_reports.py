import json
from pathlib import Path
import pandas as pd

root=Path("outputs")
rows=[]
for path in sorted(root.glob("*/test_metrics.json")):
    data=json.loads(path.read_text())
    rows.append({"model":path.parent.name,"accuracy":data.get("accuracy"),"macro_f1":data.get("macro_f1"),"weighted_f1":data.get("weighted_f1")})
if not rows: raise SystemExit("No evaluation JSON files found. Train a model first.")
out=root/"model_comparison.csv"
pd.DataFrame(rows).to_csv(out,index=False)
print(f"Wrote {out}")
