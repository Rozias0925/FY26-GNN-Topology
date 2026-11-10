"""CLI for the independent version-2 mechanism simulation study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    simulate=commands.add_parser("simulate")
    simulate.add_argument("--config",default="algorithm/configs/mechanism_v2.json")
    simulate.add_argument("--stages",nargs="+",choices=["A","B","C","D","sample_size","robustness"])
    simulate.add_argument("--output",default="algorithm/results/mechanism_v2")
    simulate.add_argument("--dry-run",action="store_true")
    aggregate=commands.add_parser("report")
    aggregate.add_argument("--output",required=True)
    freeze=commands.add_parser("freeze-expectation")
    freeze.add_argument("--panel",required=True)
    freeze.add_argument("--synthetic",required=True)
    freeze.add_argument("--output",required=True)
    freeze.add_argument("--gate-epochs",type=int,default=80)
    validate=commands.add_parser("validate-real")
    validate.add_argument("--expectation",required=True)
    validate.add_argument("--heldout",required=True)
    validate.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="simulate":
        config=json.loads(Path(args.config).read_text(encoding="utf-8"))
        stages=args.stages or config.get("stages",["A"])
        from gnn_mechanisms.experiment import workload,run
        plan=workload(config,stages)
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        if not args.dry_run:
            print(json.dumps(run(config,stages,args.output),ensure_ascii=False,indent=2))
    elif args.command=="report":
        import pandas as pd
        from gnn_mechanisms.reporting import export
        from gnn_mechanisms.experiment import read_json
        folder=Path(args.output)
        if (folder/"run.lock").exists():
            raise RuntimeError("Do not re-export while training is writing this directory.")
        config=read_json(folder/"manifest.json")["config"]
        export(pd.read_csv(folder/"metrics.csv",keep_default_na=False).replace("",float("nan")),folder,
               config.get("bootstrap_repetitions",2000),config.get("purpose","research"))
    elif args.command=="freeze-expectation":
        from gnn_mechanisms.real_validation import freeze
        payload=freeze(args.panel,args.synthetic,args.output,args.gate_epochs)
        print(json.dumps(dict(output=args.output,support=payload["support"]),ensure_ascii=False,indent=2))
    else:
        from gnn_mechanisms.real_validation import compare
        payload=compare(args.expectation,args.heldout,args.output)
        print(json.dumps(dict(output=args.output,comparisons=payload["comparisons"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
