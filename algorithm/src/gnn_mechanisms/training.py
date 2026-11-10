"""Equal candidate budgets, validation-only selection, and held-out evaluation."""
import copy
import itertools
import time
import numpy as np
import torch
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from .diagnostics import diagnostic_target
from .models import SnapshotModel, pack_graphs, graph_tensors, select_graph


def prepare(dataset,device="cpu"):
    train=dataset.splits["train"]
    x=dataset.features.astype(float)
    mean=x[train].mean((0,1))
    scale=np.maximum(x[train].std((0,1)),1e-6)
    y=diagnostic_target(dataset)
    y_mean=float(y[train].mean()) if dataset.condition.task=="regression" else 0.
    y_scale=max(float(y[train].std()),1e-6) if dataset.condition.task=="regression" else 1.
    return dict(x=torch.as_tensor((x-mean)/scale,dtype=torch.float32,device=device),
                y=torch.as_tensor((y-y_mean)/y_scale,dtype=torch.float32,device=device),
                raw_y=y,mean=mean.tolist(),scale=scale.tolist(),y_mean=y_mean,y_scale=y_scale,
                static=graph_tensors(pack_graphs(dataset.static_graph),device),
                dynamic=graph_tensors(pack_graphs(dataset.observed_graphs),device),
                splits=dataset.splits,task=dataset.condition.task,
                target_times=dataset.target_times)


def graph_for(name,data,indices):
    return select_graph(data["dynamic"],indices) if name.startswith("Dynamic") else data["static"]


def loss_function(prediction,target,task):
    if task=="classification":
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction,target)
    return torch.nn.functional.mse_loss(prediction,target)


def predict(model,data,indices,batch_size=4,scaled=False):
    model.eval()
    output=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size]
            output.append(model(data["x"][ix],graph_for(model.name,data,ix)).cpu().numpy())
    result=np.concatenate(output)
    if scaled:
        return result
    if data["task"]=="classification":
        return 1/(1+np.exp(-np.clip(result,-60,60)))
    return result*data["y_scale"]+data["y_mean"]


def fit_candidate(name,data,config,width,rate,seed):
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("threads",2))
    torch.use_deterministic_algorithms(True)
    model=SnapshotModel(name,data["x"].shape[-1],width).to(data["x"].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=rate,
                                weight_decay=config.get("weight_decay",1e-4))
    rng=np.random.default_rng(seed)
    batch_size=config.get("batch_size",4)
    best_loss=float("inf")
    best=None
    stale=0
    history=[]
    for epoch in range(config.get("epochs",200)):
        model.train()
        losses=[]
        order=rng.permutation(data["splits"]["train"])
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            prediction=model(data["x"][indices],graph_for(name,data,indices))
            loss=loss_function(prediction,data["y"][indices],data["task"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: nonfinite training loss.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid=data["splits"]["validation"]
        raw=predict(model,data,valid,batch_size,scaled=True)
        valid_loss=float(loss_function(torch.as_tensor(raw),
            data["y"][valid].cpu(),data["task"]))
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),
                            validation_loss=valid_loss))
        if valid_loss<best_loss-config.get("min_delta",1e-6):
            best_loss=valid_loss
            best=copy.deepcopy({key:value.detach().cpu() for key,value in model.state_dict().items()})
            best_epoch=epoch+1
            stale=0
        else:
            stale+=1
        if stale>=config.get("patience",25):
            break
    if best is None:
        raise RuntimeError("No finite validation checkpoint.")
    model.load_state_dict(best)
    limit=config.get("epochs",200)
    return model,dict(width=width,learning_rate=rate,best_epoch=best_epoch,
                      validation_loss=best_loss,history=history,
                      epochs_run=len(history),max_epochs=limit,
                      stopped_reason="early_stopping" if stale>=config.get("patience",25) else "epoch_limit",
                      budget_exhausted=bool(len(history)==limit and stale<config.get("patience",25)),
                      parameter_count=sum(p.numel() for p in model.parameters()))


def metrics(target,prediction,task):
    if task=="regression":
        difference=prediction-target
        return dict(r2=float(r2_score(target.ravel(),prediction.ravel())),
                    rmse=float(np.sqrt(np.mean(difference**2))),
                    mae=float(np.mean(np.abs(difference))))
    y=target.ravel()
    p=np.clip(prediction.ravel(),1e-7,1-1e-7)
    both=len(np.unique(y))==2
    return dict(ap=float(average_precision_score(y,p)) if both else None,
                auroc=float(roc_auc_score(y,p)) if both else None,
                brier=float(np.mean((p-y)**2)),
                log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                prevalence=float(y.mean()),single_class=not both)


def train_model(name,data,config,seed):
    started=time.perf_counter()
    if config.get("epochs",200)<1 or config.get("batch_size",4)<1:
        raise ValueError("Invalid training budget.")
    grid=list(itertools.product(config.get("widths",[32,64]),
                                config.get("learning_rates",[.001,.003])))
    if not grid:
        raise ValueError("Empty candidate grid.")
    candidates=[]
    best=None
    best_meta=None
    for width,rate in grid:
        if width<2 or rate<=0:
            raise ValueError("Invalid candidate.")
        model,meta=fit_candidate(name,data,config,width,rate,seed)
        candidates.append(meta)
        if best_meta is None or meta["validation_loss"]<best_meta["validation_loss"]:
            best,best_meta=model,meta
    # The test data is first scored after the candidate and epoch are selected.
    validation=data["splits"]["validation"]
    test=data["splits"]["test"]
    vp=predict(best,data,validation,config.get("batch_size",4))
    tp=predict(best,data,test,config.get("batch_size",4))
    result=dict(model=name,training_seed=seed,task=data["task"],
                selection={key:value for key,value in best_meta.items() if key!="history"},
                validation=metrics(data["raw_y"][validation],vp,data["task"]),
                test=metrics(data["raw_y"][test],tp,data["task"]),
                training_seconds=time.perf_counter()-started,
                candidates=len(grid),feature_mean=data["mean"],feature_scale=data["scale"],
                response_mean=data["y_mean"],response_scale=data["y_scale"],
                validation_target_times=data["target_times"][validation].tolist(),
                test_target_times=data["target_times"][test].tolist())
    predictions=dict(validation_prediction=vp,test_prediction=tp,
                     validation_target=data["raw_y"][validation],
                     test_target=data["raw_y"][test],
                     validation_target_times=data["target_times"][validation],
                     test_target_times=data["target_times"][test])
    return result,predictions,best.state_dict(),candidates
