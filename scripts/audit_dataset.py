import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from PIL import Image

CLASSES=["glioma","meningioma","notumor","pituitary"]
VALID_EXTENSIONS={".jpg",".jpeg",".png",".bmp"}

def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description="Audit MRI images and exact duplicate leakage.")
    parser.add_argument("--data-dir",required=True)
    parser.add_argument("--output-dir",default="outputs/audit")
    args=parser.parse_args()
    root=Path(args.data_dir).resolve()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; errors=[]
    for split in ("Training","Testing"):
        for class_name in CLASSES:
            folder=root/split/class_name
            if not folder.exists():
                errors.append(f"Missing folder: {folder}")
                continue
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS: continue
                try:
                    file_hash=sha256(path)
                    with Image.open(path) as image: image.verify()
                    with Image.open(path) as image:
                        width,height,mode,fmt=image.width,image.height,image.mode,image.format
                    rows.append({"path":str(path),"original_split":split,"partition":"train" if split=="Training" else "test","class_name":class_name,"label_id":CLASSES.index(class_name),"sha256":file_hash,"width":width,"height":height,"mode":mode,"format":fmt})
                except Exception as exc: errors.append(f"{path}: {exc}")
    if not rows:
        raise SystemExit("No readable images found. Check the data directory layout.")
    df=pd.DataFrame(rows)
    grouped=df.groupby("sha256")
    df["label_conflict"]=grouped.class_name.transform("nunique")>1
    df["has_test_copy"]=grouped.partition.transform(lambda values:(values=="test").any())
    df=df.sort_values(["sha256","has_test_copy","partition","path"],ascending=[True,False,True,True])
    df["keep"]=~df.duplicated("sha256",keep="first")
    # Keep one representative per exact hash; if any copy was in Testing, assign the hash to test.
    df["partition"]=df.has_test_copy.map({True:"test",False:"train"})
    df["usable"]=df.keep & ~df.label_conflict
    root.mkdir(parents=True,exist_ok=True)
    df.to_csv(root/"manifest.csv",index=False)
    duplicate_rows=df[df.sha256.isin(df.sha256[df.duplicated("sha256",keep=False)])]
    duplicate_rows.to_csv(output/"exact_duplicates.csv",index=False)
    df.groupby(["original_split","class_name"]).size().rename("count").reset_index().to_csv(output/"class_counts_raw.csv",index=False)
    df[df.usable].groupby(["partition","class_name"]).size().rename("unique_usable_count").reset_index().to_csv(output/"class_counts_deduplicated.csv",index=False)
    groups=df.groupby("sha256")
    summary={"total_valid_images":len(df),"unique_hashes":int(df.sha256.nunique()),"duplicate_groups":int((groups.size()>1).sum()),"cross_split_duplicate_groups":int(groups.original_split.nunique().gt(1).sum()),"label_conflict_groups":int(groups.label_conflict.first().sum()),"dimensions_top":df.groupby(["width","height"]).size().sort_values(ascending=False).head(15).reset_index(name="count").to_dict("records"),"errors":errors}
    (output/"audit_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
