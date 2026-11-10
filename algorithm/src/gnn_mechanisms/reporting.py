"""Paired contrasts, seed bootstrap, response curves, and observable phase maps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONTRASTS={
    "Static-GCN_minus_MLP":{"Static-GCN":1,"MLP":-1},
    "Static-GAT_minus_MLP":{"Static-GAT":1,"MLP":-1},
    "Dynamic-GCN_minus_MLP":{"Dynamic-GCN":1,"MLP":-1},
    "Dynamic-GAT_minus_MLP":{"Dynamic-GAT":1,"MLP":-1},
    "Architecture_Static":{"Static-GAT":1,"Static-GCN":-1},
    "Architecture_Dynamic":{"Dynamic-GAT":1,"Dynamic-GCN":-1},
    "Dynamics_GCN":{"Dynamic-GCN":1,"Static-GCN":-1},
    "Dynamics_GAT":{"Dynamic-GAT":1,"Static-GAT":-1},
    "Interaction":{"Dynamic-GAT":1,"Static-GAT":-1,"Dynamic-GCN":-1,"Static-GCN":1},
}
ROBUSTNESS_CONTRASTS={
    "GATv2_minus_GAT_Static":{"Static-GATv2":1,"Static-GAT":-1},
    "GATv2_minus_GAT_Dynamic":{"Dynamic-GATv2":1,"Dynamic-GAT":-1},
    "Dynamics_GATv2":{"Dynamic-GATv2":1,"Static-GATv2":-1},
}
DIAGNOSTICS=["S_X","S_G","S_G_shuffled","H","R_hat","R_true","R_estimation_error","D_struct","D_pred"]


def bootstrap(values,repetitions=2000,seed=47):
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired differences are required.")
    means=np.random.default_rng(seed).choice(values,(repetitions,len(values)),replace=True).mean(1)
    return dict(mean_gap=float(values.mean()),
                ci_low=float(np.quantile(means,.025)) if len(values)>1 else None,
                ci_high=float(np.quantile(means,.975)) if len(values)>1 else None,
                p_delta_positive=float((values>0).mean()),
                bootstrap_mean_positive=float((means>0).mean()),seeds=len(values),
                ci_scope="paired resampling of independent DATASET seeds",
                inference_ready=len(values)>=5)


def paired(frame,repetitions=2000):
    frame=frame.copy()
    # CSV reloads with preserved "null" axis labels can give mixed score
    # columns a string dtype (especially regression/classification together).
    for column in DIAGNOSTICS+["test_r2","test_ap"]:
        if column in frame:
            frame[column]=pd.to_numeric(frame[column],errors="coerce")
    output=[]
    for (stage,axis,value,condition_id),group in frame.groupby(
            ["stage","axis","value","condition_id"],dropna=False,sort=False):
        task=group["task"].iloc[0]
        primary="test_r2" if task=="regression" else "test_ap"
        wide=group.pivot(index="seed",columns="model",values=primary)
        definitions={**CONTRASTS,**ROBUSTNESS_CONTRASTS}
        for contrast,coefficients in definitions.items():
            if not set(coefficients)<=set(wide.columns):
                continue
            relevant=wide[list(coefficients)].dropna()
            if len(relevant)!=len(wide):
                continue
            gaps=sum(relevant[name]*weight for name,weight in coefficients.items())
            row=dict(stage=stage,axis=axis,value=value,condition_id=condition_id,
                     task=task,primary_score=primary,contrast=contrast,
                     **bootstrap(gaps.to_numpy(),repetitions))
            for diagnostic in DIAGNOSTICS:
                row[diagnostic]=float(group.groupby("seed")[diagnostic].first().mean())
            output.append(row)
    return pd.DataFrame(output,columns=list(output[0]) if output else
        ["stage","axis","value","condition_id","task","primary_score","contrast",
         "mean_gap","ci_low","ci_high","p_delta_positive","seeds",*DIAGNOSTICS])


def figures(frame,contrasts,folder,purpose="research"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    axis_diagnostic={"graph_signal":"S_G","kappa":"R_true","tau":"H",
                     "delta":"D_pred","eta":"D_struct"}
    axis_contrasts={
        "graph_signal":["Static-GCN_minus_MLP","Static-GAT_minus_MLP"],
        "kappa":["Architecture_Static","Architecture_Dynamic"],
        "tau":["Static-GCN_minus_MLP","Static-GAT_minus_MLP"],
        "delta":["Dynamics_GCN","Dynamics_GAT"],
        "eta":["Dynamics_GCN","Dynamics_GAT"],
    }
    paths=[]
    for axis,diagnostic in axis_diagnostic.items():
        subset=contrasts[(contrasts.stage=="B")&(contrasts.axis==axis)]
        if subset.empty:
            continue
        for task,task_frame in subset.groupby("task"):
            fig,axs=plt.subplots(1,2,figsize=(10,4))
            for contrast in axis_contrasts[axis]:
                points=task_frame[task_frame.contrast==contrast].copy()
                points["knob"]=pd.to_numeric(points.value)
                points=points.sort_values("knob")
                for ax,key in zip(axs,("knob",diagnostic)):
                    finite=points.dropna(subset=[key,"mean_gap"])
                    if finite.empty:
                        continue
                    if finite[["ci_low","ci_high"]].isna().any().any():
                        ax.plot(finite[key],finite.mean_gap,marker="o",label=contrast)
                        continue
                    errors=np.vstack((finite.mean_gap-finite.ci_low,
                                      finite.ci_high-finite.mean_gap))
                    ax.errorbar(finite[key],finite.mean_gap,yerr=np.maximum(errors,0),
                                marker="o",capsize=3,label=contrast)
            for ax,label in zip(axs,(axis,diagnostic)):
                ax.axhline(0,color="gray",linewidth=.7)
                ax.set_xlabel(label+(" (oracle)" if label=="R_true" else ""))
                ax.set_ylabel("Paired score difference")
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
            prefix="SMOKE — execution check only: " if purpose=="smoke" else ""
            count=int(task_frame.seeds.min())
            fig.suptitle(f"{prefix}{axis}: {task}; {count} seeds",fontsize=11)
            fig.tight_layout()
            path=folder/f"response_{axis}_{task}.png"
            fig.savefig(path,dpi=160)
            plt.close(fig)
            paths.append(path.name)
    phase_specs=[("S_G","R_hat","Architecture_Static"),
                 ("D_struct","D_pred","Dynamics_GCN"),
                 ("S_G","H","Static-GCN_minus_MLP")]
    for x,y,contrast in phase_specs:
        points=contrasts[(contrasts.stage.isin(["C","D"]))&
                        (contrasts.contrast==contrast)&(contrasts.task=="regression")]
        points=points.drop_duplicates("condition_id").dropna(subset=[x,y,"mean_gap"])
        if points.empty:
            continue
        fig,ax=plt.subplots(figsize=(6,4))
        limit=max(float(points.mean_gap.abs().max()),1e-4)
        scatter=ax.scatter(points[x],points[y],c=points.mean_gap,cmap="RdBu_r",
                           vmin=-limit,vmax=limit,s=35,edgecolors="gray",linewidths=.3)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        prefix="SMOKE — execution check only\n" if purpose=="smoke" else ""
        ax.set_title(prefix+contrast+"; observable diagnostics",fontsize=10)
        fig.colorbar(scatter,ax=ax,label="Mean paired score difference")
        fig.tight_layout()
        path=folder/f"phase_{x}_{y}_{contrast}.png"
        fig.savefig(path,dpi=160)
        plt.close(fig)
        paths.append(path.name)
    # Report sample efficiency and robustness, including classification AP.
    for stage in ("sample_size","robustness"):
        subset=contrasts[contrasts.stage==stage]
        for (axis,task),group in subset.groupby(["axis","task"]):
            fig,ax=plt.subplots(figsize=(8,4))
            labels=list(dict.fromkeys(group.value.astype(str)))
            numeric=pd.to_numeric(pd.Series(labels),errors="coerce")
            if numeric.notna().all():
                labels=[labels[i] for i in np.argsort(numeric.to_numpy())]
            for contrast in ("Static-GCN_minus_MLP","Static-GAT_minus_MLP",
                             "Architecture_Static","GATv2_minus_GAT_Static"):
                points=group[group.contrast==contrast].copy()
                if points.empty:
                    continue
                points["position"]=points.value.astype(str).map({v:i for i,v in enumerate(labels)})
                points=points.sort_values("position")
                ready=points.dropna(subset=["mean_gap","ci_low","ci_high"])
                if ready.empty:
                    ax.scatter(points.position,points.mean_gap,label=contrast)
                else:
                    errors=np.vstack((ready.mean_gap-ready.ci_low,ready.ci_high-ready.mean_gap))
                    ax.errorbar(ready.position,ready.mean_gap,yerr=np.maximum(errors,0),
                                marker="o",capsize=3,label=contrast)
            ax.axhline(0,color="gray",linewidth=.7)
            ax.set_xticks(range(len(labels)),labels,rotation=20 if len(labels)>4 else 0)
            ax.set_xlabel(axis)
            ax.set_ylabel("Paired AP difference" if task=="classification" else "Paired R2 difference")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            prefix="SMOKE — execution check only: " if purpose=="smoke" else ""
            ax.set_title(f"{prefix}{stage}: {axis}; {int(group.seeds.min())} seeds",fontsize=10)
            fig.tight_layout()
            path=folder/f"{stage}_{axis}_{task}.png"
            fig.savefig(path,dpi=160)
            plt.close(fig)
            paths.append(path.name)
    return paths


def export(frame,folder,repetitions=2000,purpose="research"):
    folder=Path(folder)
    frame=frame.copy()
    numeric=DIAGNOSTICS+[column for column in frame if column.startswith("test_")
                         and column!="test_single_class"]
    for column in numeric:
        if column in frame:
            frame[column]=pd.to_numeric(frame[column],errors="coerce")
    if "test_single_class" in frame:
        frame["test_single_class"]=frame["test_single_class"].map(
            lambda value:None if pd.isna(value) or str(value) in ("","nan","None")
            else str(value).lower()=="true")
    frame.to_csv(folder/"metrics.csv",index=False)
    contrast_frame=paired(frame,repetitions)
    contrast_frame.to_csv(folder/"paired_contrasts.csv",index=False)
    contrast_frame[contrast_frame.stage=="B"].to_csv(folder/"response_curves.csv",index=False)
    contrast_frame[contrast_frame.stage.isin(["C","D"])].to_csv(folder/"phase_points.csv",index=False)
    figures_list=figures(frame,contrast_frame,folder/"figures",purpose)
    clean=lambda data:json.loads(data.to_json(orient="records"))
    payload=dict(schema_version=2,purpose=purpose,primary_evidence="paired response curves",
                 probability_definition="p_delta_positive = fraction of dataset seeds with positive gap",
                 caution="Smoke outputs verify execution, not directional hypotheses." if purpose=="smoke"
                         else "Intervals use dataset seeds; few seeds imply imprecise uncertainty.",
                 model_results=clean(frame),contrasts=clean(contrast_frame),figures=figures_list)
    (folder/"website_results_v2.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload
