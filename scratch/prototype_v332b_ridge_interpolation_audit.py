"""v332b: auditoría float64 de la cresta de interpolación observada en v332."""
from __future__ import annotations
import argparse, json, math, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
import prototype_v332_spectral_sample_efficiency as v332

@dataclass(frozen=True)
class Config:
    experiment_id:str; mode:str; rigor_level:int; seeds:tuple[int,...]; sample_sizes:tuple[int,...]; ridges:tuple[float,...]; test_examples:int; device:str

class Log:
    def __init__(self): self.t=time.perf_counter()
    def line(self,s:str):
        e=time.perf_counter()-self.t; h,r=divmod(e,3600); m,s2=divmod(r,60)
        for x in str(s).splitlines() or [""]: print(f"[+{int(h):02d}:{int(m):02d}:{s2:05.2f}] {x}",flush=True)
    def obj(self,k:str,x:Any): self.line(k+":"); self.line(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,default=str))

def parse()->argparse.Namespace:
    p=argparse.ArgumentParser(description="v332b ridge interpolation audit"); p.add_argument("--mode",choices=("pilot","level2"),default="pilot"); p.add_argument("--device",default="cpu"); p.add_argument("--run-id",default="v332b_ridge_interpolation_audit"); p.add_argument("--seeds",default=None); return p.parse_args()
def build(a:argparse.Namespace)->Config:
    level,seeds,test=(1,(42,),2048) if a.mode=="pilot" else (2,(10,20,30,42,100),8192)
    if a.seeds: seeds=tuple(int(x) for x in a.seeds.split(","))
    return Config(a.run_id,a.mode,level,seeds,(48,64,80,128),(1e-5,1e-4,1e-3,1e-2),test,a.device)
def stats(x:list[float])->dict[str,float]:
    m,sd,se=v332.mean_se(x); return {"mean":m,"sd":sd,"se":se}
def fit64(student:str,x:torch.Tensor,y:torch.Tensor,b:torch.Tensor|None,ridge:float)->torch.Tensor:
    if student=="dense_linear": return torch.linalg.solve(x.T@x+ridge*torch.eye(v332.DIM,dtype=torch.float64),x.T@y)
    z=x@b.T; t=y@b.T; return (z*t).sum(0)/(z.square().sum(0)+ridge)
def pred64(student:str,p:torch.Tensor,x:torch.Tensor,b:torch.Tensor|None)->torch.Tensor: return x@p if student=="dense_linear" else ((x@b.T)*p)@b
def main()->int:
    log=Log()
    try:
        a=parse(); c=build(a); device=torch.device(c.device)
        if device.type!="cpu": raise ValueError("v332b usa CPU/float64 para auditar condicionamiento")
        start=datetime.now(timezone.utc).isoformat(); log.line("="*96); log.line(f"EXPERIMENT_START | {c.experiment_id} — ridge/interpolación float64"); log.line("QUESTION | ¿El pico Dense en n≈64 cambia sistemáticamente con ridge?"); log.obj("METADATA",{"started_utc":start,"argv":sys.argv,**v332.metadata(device)}); log.obj("FULL_CONFIG",asdict(c)); log.line("ARCHITECTURE | dense=4096 parámetros; diagonales=64; ajuste/predicción float64; sin épocas por solución cerrada.")
        rows=[]
        for seed in c.seeds:
            for teacher in v332.TEACHERS:
                w,tm=v332.teacher_matrix(teacher,seed); w=w.double(); g=torch.Generator().manual_seed(334000+seed); xt=torch.randn(c.test_examples,v332.DIM,generator=g,dtype=torch.float64); yt=xt@w+v332.NOISE_STD*torch.randn(c.test_examples,v332.DIM,generator=g,dtype=torch.float64)
                log.obj("TEACHER",{"seed":seed,"teacher":teacher,**tm})
                for n in c.sample_sizes:
                    gx=torch.Generator().manual_seed(335000+seed*100+n+v332.TEACHERS.index(teacher)); x=torch.randn(n,v332.DIM,generator=gx,dtype=torch.float64); y=x@w+v332.NOISE_STD*torch.randn(n,v332.DIM,generator=gx,dtype=torch.float64)
                    for ridge in c.ridges:
                        gram=x.T@x+ridge*torch.eye(v332.DIM,dtype=torch.float64); cond=torch.linalg.cond(gram).item()
                        for student in v332.STUDENTS:
                            b,bm=v332.basis_for_student(student,seed); b=None if b is None else b.double(); t0=time.perf_counter(); p=fit64(student,x,y,b,ridge); sec=time.perf_counter()-t0; train=(pred64(student,p,x,b)-y).square().mean().item(); test=(pred64(student,p,xt,b)-yt).square().mean().item(); r={"seed":seed,"teacher":teacher,"n_train":n,"ridge":ridge,"student":student,"params":p.numel(),"basis":bm,"condition_number":cond,"train_mse":train,"test_mse":test,"fit_seconds":sec}; rows.append(r); log.line(f"CONDITION | seed={seed} | teacher={teacher} | n={n} | ridge={ridge:.0e} | student={student} | params={p.numel()} | gram_condition={cond:.3e} | train_mse={train:.8f} | test_mse={test:.8f} | fit_seconds={sec:.5f}")
        summary={}
        for teacher in v332.TEACHERS:
            for n in c.sample_sizes:
                for ridge in c.ridges:
                    for student in v332.STUDENTS:
                        rs=[r for r in rows if r["teacher"]==teacher and r["n_train"]==n and r["ridge"]==ridge and r["student"]==student]; summary[f"{teacher}|n={n}|ridge={ridge:g}|{student}"]={"teacher":teacher,"n_train":n,"ridge":ridge,"student":student,"n_seeds":len(rs),"test_mse":stats([r["test_mse"] for r in rs]),"condition_number":stats([r["condition_number"] for r in rs])}
        log.obj("SUMMARY",summary); payload={"experiment_id":c.experiment_id,"started_utc":start,"metadata":v332.metadata(device),"config":asdict(c),"question":"¿El pico Dense en n≈64 cambia sistemáticamente con ridge?","results":rows,"summary":summary}; v332.RAW_DIR.mkdir(parents=True,exist_ok=True); out=v332.RAW_DIR/f"{c.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"; out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8"); log.line(f"ARTIFACT | raw_results={out}"); log.line("EXPERIMENT_COMPLETE | status=success | next_step=inspeccionar_JSON_antes_de_redactar_findings"); return 0
    except Exception as e: log.line(f"EXPERIMENT_ERROR | type={type(e).__name__} | message={e}"); return 1
if __name__=="__main__": raise SystemExit(main())
