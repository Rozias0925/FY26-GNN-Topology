"""Audit production-size V2 generators and training-only probes, without GNN fits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).parent/"src"))
from gnn_mechanisms.config import plans
from gnn_mechanisms.data import generate
from gnn_mechanisms.diagnostics import measure
from gnn_mechanisms.experiment import write_json,source_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="algorithm/configs/mechanism_v2.json")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--output",default="algorithm/results/v2_generator_review.json")
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():
        raise FileExistsError("Preserve the previous generator audit; use another filename.")
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    report=dict(purpose="generator_and_probe_audit",trained_gnn_models=0,
                source_sha256=source_hash(),seed=args.seed,config=config,
                created_at=datetime.now(timezone.utc).isoformat(),conditions=[])
    for entry in plans(config,["A"]):
        started=time.perf_counter()
        condition=entry["condition"]
        dataset=generate(condition,args.seed)
        diagnostics=measure(dataset,config.get("diagnostics",{}))
        result=dict(regime=entry["axis"],condition=condition.to_dict(),oracle=dataset.oracle,
                    diagnostics=diagnostics,seconds=time.perf_counter()-started)
        report["conditions"].append(result)
        print(entry["axis"],json.dumps({key:diagnostics[key] for key in
            ("S_X","S_G","H","R_hat","R_true","D_struct","D_pred")}),flush=True)
    write_json(destination,report)
    print(str(destination),flush=True)


if __name__=="__main__":
    main()
