"""Prepare matched 20-epoch recipes from the completed low-light preview."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from car_smart_assist.config.detection import load_yolo_config


def save(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-run",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    run,output=args.preview_run.resolve(),args.output_dir.resolve()
    terminal=json.loads((run/"terminal.json").read_bytes())
    scan=json.loads((run/"preview/train_statistics.json").read_bytes())
    cfg=load_yolo_config(ROOT/"configs/model/yolo_detection.yaml")
    manifest=json.loads(((ROOT/cfg["data"]).parent/"manifest.json").read_bytes())
    train={r["image"]:r for r in manifest["images"] if r["split"]=="train"}
    if (terminal["status"]!="COMPLETED" or terminal["changed_files"] or scan["split"]!="train"
            or scan["dataset_signature"]!=manifest["signature"]
            or len(scan["records"])!=len(train) or {r["image"] for r in scan["records"]}!=set(train)):
        raise ValueError("Require complete immutable training preview evidence")
    eligible={r["image"]:r["mean"] for r in scan["records"] if train[r["image"]]["condition"]!="night" and r["mean"]>=80}
    ordered=sorted(eligible)
    seed=cfg["train"]["seed"]
    rng=np.random.default_rng(seed)
    schedule=[sorted(rng.choice(ordered,size=len(ordered)//4,replace=False).tolist()) for _ in range(20)]
    if len(eligible)!=1746:
        raise ValueError("Eligible input population changed since review")
    initial=cfg["checkpoint"]
    weight_hash=hashlib.sha256((ROOT/initial).read_bytes()).hexdigest()
    plan={"version":1,"dataset_signature":manifest["signature"],"epochs":20,"imgsz":960,"seed":seed,
          "minimum_mean":80,"exposure":.5,"shot_scale":.001,"read_std":.001,
          "initial_sha256":weight_hash,"eligible":eligible,"schedule":schedule}
    output.mkdir(parents=True,exist_ok=False)
    path=output/"plan.json"
    save(path,plan)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for mode in ("control","lowlight"):
        arm=deepcopy(cfg)
        arm.update(model=initial,project=str(output/"models"),name=mode,device="cuda:0",
                   checkpoint=str(output/"models"/mode/"weights/best.pt"),
                   calibration_file=str(output/"calibration"/mode/"calibration.json"),
                   training_lowlight={"mode":mode,"plan":str(path),"sha256":digest})
        arm["train"].update(epochs=20,patience=0,amp=False)
        with (output/(mode+".yaml")).open("x",encoding="utf-8") as stream:
            yaml.safe_dump({"yolo_detection":arm},stream,sort_keys=False,allow_unicode=True)
    save(output/"recipe.json",{"eligible":len(eligible),"per_epoch":len(ordered)//4,
          "epochs_per_arm":20,"planned_exposures":sum(map(len,schedule)),"initial_weight":initial,
          "initial_sha256":weight_hash,"plan_sha256":digest})
    print("Prepared 20+20 epochs;",len(eligible),"eligible frames;",sum(map(len,schedule)),"planned exposures")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
