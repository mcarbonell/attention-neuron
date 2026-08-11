"""v332: teacher-student de eficiencia muestral y longitud de descripción.

Soluciones cerradas ridge; no hay épocas porque no hay optimización iterativa.
Cada línea de salida tiene tiempo relativo y el JSON conserva todas las curvas.
"""
from __future__ import annotations

import argparse, hashlib, json, math, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR, LEDGER = ROOT / "results" / "raw", ROOT / "results" / "master_ledger.jsonl"
DIM, ACTIVE, NOISE_STD, RIDGE = 64, 8, 0.05, 1e-4
TEACHERS, STUDENTS = ("dct_sparse", "random_sparse", "dense_full"), ("dense_linear", "dct_diagonal", "random_diagonal")

@dataclass(frozen=True)
class Config:
    experiment_id: str; mode: str; rigor_level: int; seeds: tuple[int, ...]; sample_sizes: tuple[int, ...]
    test_examples: int; device: str; deterministic: bool; dim: int = DIM; active_modes: int = ACTIVE
    noise_std: float = NOISE_STD; ridge: float = RIDGE; quant_bits: tuple[int, ...] = (4, 8, 16)

class Log:
    def __init__(self): self.start = time.perf_counter()
    def line(self, msg: str):
        e=time.perf_counter()-self.start; h,r=divmod(e,3600); m,s=divmod(r,60)
        for x in str(msg).splitlines() or [""]: print(f"[+{int(h):02d}:{int(m):02d}:{s:05.2f}] {x}", flush=True)
    def obj(self, name: str, obj: Any): self.line(name+":"); self.line(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str))

def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="v332 spectral sample efficiency")
    p.add_argument("--mode", choices=("pilot","level2"), default="pilot"); p.add_argument("--device",default="cpu")
    p.add_argument("--run-id",default="v332_spectral_sample_efficiency"); p.add_argument("--seeds",default=None)
    p.add_argument("--no-deterministic",action="store_true"); return p.parse_args()

def config(a: argparse.Namespace) -> Config:
    base={"pilot":(1,(42,),(4,16,64),2048),"level2":(2,(10,20,30,42,100),(4,8,16,32,64,128),8192)}[a.mode]
    seeds=tuple(int(x) for x in a.seeds.split(",")) if a.seeds else base[1]
    return Config(a.run_id,a.mode,base[0],seeds,base[2],base[3],a.device,not a.no_deterministic)

def sha(x: torch.Tensor) -> str: return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def dct(dim: int) -> torch.Tensor:
    i=torch.arange(dim,dtype=torch.float64).unsqueeze(1); j=torch.arange(dim,dtype=torch.float64).unsqueeze(0)
    u=torch.cos(math.pi/dim*(i+0.5)*j); u[:,0]/=math.sqrt(dim); u[:,1:]*=math.sqrt(2/dim); return u.float()
def orthogonal(dim: int, seed: int) -> torch.Tensor:
    g=torch.Generator().manual_seed(seed); q,r=torch.linalg.qr(torch.randn(dim,dim,generator=g)); return q*torch.sign(torch.diag(r)).unsqueeze(0)
def teacher_matrix(name: str, seed: int) -> tuple[torch.Tensor, dict[str,Any]]:
    if name=="dct_sparse": b=dct(DIM); label="dct2"; bseed=None
    elif name=="random_sparse": b=orthogonal(DIM, 332000+seed); label="random_q"; bseed=332000+seed
    else:
        g=torch.Generator().manual_seed(333000+seed); w=torch.randn(DIM,DIM,generator=g)/math.sqrt(DIM); return w,{"kind":"dense_full","seed":333000+seed,"sha256":sha(w)}
    a=torch.zeros(DIM); a[:ACTIVE]=torch.linspace(0.4,1.2,ACTIVE)
    w=b.t()@torch.diag(a)@b
    return w,{"kind":name,"basis":label,"basis_seed":bseed,"basis_sha256":sha(b),"active_modes":ACTIVE,"sha256":sha(w)}
def basis_for_student(name: str, seed: int) -> tuple[torch.Tensor|None,dict[str,Any]]:
    if name=="dense_linear": return None,{"basis":"none"}
    if name=="dct_diagonal": b=dct(DIM); return b,{"basis":"dct2","sha256":sha(b)}
    b=orthogonal(DIM,332000+seed); return b,{"basis":"random_q","seed":332000+seed,"sha256":sha(b)}
def predict(student: str, fitted: torch.Tensor, x: torch.Tensor, basis: torch.Tensor|None) -> torch.Tensor:
    return x@fitted if student=="dense_linear" else ((x@basis.t())*fitted)@basis
def fit(student: str, x: torch.Tensor, y: torch.Tensor, basis: torch.Tensor|None) -> torch.Tensor:
    if student=="dense_linear": return torch.linalg.solve(x.t()@x+RIDGE*torch.eye(DIM),x.t()@y)
    z=x@basis.t(); t=y@basis.t(); return (z*t).sum(0)/(z.square().sum(0)+RIDGE)
def quantize(v: torch.Tensor, bits: int) -> torch.Tensor:
    lo,hi=v.min(),v.max(); levels=2**bits-1
    if (hi-lo).abs()<1e-12: return v.clone()
    return (torch.round((v-lo)/(hi-lo)*levels)/levels)*(hi-lo)+lo
def mean_se(vals:list[float])->tuple[float,float,float]:
    t=torch.tensor(vals,dtype=torch.float64); sd=0.0 if len(t)==1 else t.std(unbiased=True).item(); return t.mean().item(),sd,sd/math.sqrt(len(t))
def metadata(device:torch.device)->dict[str,Any]:
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit="unavailable"
    return {"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"cpu":platform.processor() or "unavailable","device":str(device),"commit_hash":commit,"torch_threads":torch.get_num_threads()}

def main()->int:
    log=Log()
    try:
        started=datetime.now(timezone.utc).isoformat(); a=args(); c=config(a); device=torch.device(c.device)
        if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA solicitada no disponible")
        log.line("="*96); log.line(f"EXPERIMENT_START | {c.experiment_id} — Eficiencia muestral espectral")
        log.line("QUESTION | ¿La base coincidente requiere menos muestras y bits para una función que Dense también representa?")
        log.line("CLAIM_BOUNDARY | Compara sesgo inductivo/MDL condicionado al teacher; no VC ni capacidad universal igual.")
        log.obj("METADATA",{"started_utc":started,"argv":sys.argv,**metadata(device)}); log.obj("FULL_CONFIG",asdict(c))
        log.line("ARCHITECTURE | dense_linear=64x64 libre (4096 parámetros); diagonales=64 ganancias + base fija (64 parámetros entrenables).")
        log.line("PROTOCOL | solución ridge cerrada: no hay épocas/steps; cada CONDITION contiene train/test y coste separados.")
        rows=[]
        for seed in c.seeds:
            torch.manual_seed(seed)
            for teacher in TEACHERS:
                w,tmeta=teacher_matrix(teacher,seed); w=w.to(device)
                g=torch.Generator().manual_seed(334000+seed); xtest=torch.randn(c.test_examples,DIM,generator=g).to(device); ytest=xtest@w+ c.noise_std*torch.randn(c.test_examples,DIM,generator=g).to(device)
                log.obj("TEACHER",{"seed":seed,"teacher":teacher,**tmeta,"test_examples":c.test_examples})
                for n in c.sample_sizes:
                    gx=torch.Generator().manual_seed(335000+seed*100+n+TEACHERS.index(teacher)); x=torch.randn(n,DIM,generator=gx).to(device); y=x@w+c.noise_std*torch.randn(n,DIM,generator=gx).to(device)
                    for student in STUDENTS:
                        b,bmeta=basis_for_student(student,seed); b=None if b is None else b.to(device)
                        t0=time.perf_counter(); fitted=fit(student,x,y,b); fit_seconds=time.perf_counter()-t0
                        train_mse=(predict(student,fitted,x,b)-y).square().mean().item(); test_mse=(predict(student,fitted,xtest,b)-ytest).square().mean().item()
                        q={};
                        for bits in c.quant_bits: q[str(bits)]={"description_bits":int(fitted.numel()*bits+32),"test_mse":(predict(student,quantize(fitted,bits),xtest,b)-ytest).square().mean().item()}
                        row={"seed":seed,"teacher":teacher,"n_train":n,"student":student,"params":int(fitted.numel()),"basis":bmeta,"train_mse":train_mse,"test_mse":test_mse,"fit_seconds":fit_seconds,"quantization":q}
                        rows.append(row); log.line(f"CONDITION | seed={seed} | teacher={teacher} | n={n} | student={student} | params={fitted.numel()} | train_mse={train_mse:.8f} | test_mse={test_mse:.8f} | fit_seconds={fit_seconds:.5f} | bits4_mse={q['4']['test_mse']:.8f} | bits4_total={q['4']['description_bits']}")
        summaries={}
        for teacher in TEACHERS:
            for n in c.sample_sizes:
                for student in STUDENTS:
                    rs=[r for r in rows if r["teacher"]==teacher and r["n_train"]==n and r["student"]==student]; m,sd,se=mean_se([r["test_mse"] for r in rs]); summaries[f"{teacher}|n={n}|{student}"]={"teacher":teacher,"n_train":n,"student":student,"n_seeds":len(rs),"test_mse_mean":m,"test_mse_sd":sd,"test_mse_se":se,"params":rs[0]["params"]}
        paired={}
        for teacher in TEACHERS:
            for n in c.sample_sizes:
                for left,right in (("dct_diagonal","random_diagonal"),("dct_diagonal","dense_linear"),("random_diagonal","dense_linear")):
                    arows={r["seed"]:r for r in rows if r["teacher"]==teacher and r["n_train"]==n and r["student"]==left}; brows={r["seed"]:r for r in rows if r["teacher"]==teacher and r["n_train"]==n and r["student"]==right}; seeds=sorted(set(arows)&set(brows)); m,sd,se=mean_se([arows[s]["test_mse"]-brows[s]["test_mse"] for s in seeds]); paired[f"{teacher}|n={n}|{left}_minus_{right}"]={"teacher":teacher,"n_train":n,"left":left,"right":right,"seeds":seeds,"mean_delta":m,"paired_sd":sd,"paired_se":se,"two_se":2*se}
        log.obj("SUMMARY",summaries); log.obj("PAIRED_COMPARISONS",paired)
        payload={"experiment_id":c.experiment_id,"started_utc":started,"metadata":metadata(device),"config":asdict(c),"question":"¿La base coincidente requiere menos muestras y bits para una función que Dense también representa?","teacher_definition":"y=x@W_star+noise; sparse W_star=B^T diag(a) B","results":rows,"summary":summaries,"paired_comparisons":paired}
        RAW_DIR.mkdir(parents=True,exist_ok=True); out=RAW_DIR/f"{c.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"; out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8"); log.line(f"ARTIFACT | raw_results={out}")
        primary=summaries[f"dct_sparse|n={max(c.sample_sizes)}|dct_diagonal"]; entry={"experiment_id":c.experiment_id,"fecha":datetime.now(timezone.utc).date().isoformat(),"familia":"spectral_sample_efficiency","dataset":"synthetic Gaussian teacher-student; fixed held-out test","n_eval":c.test_examples,"metric_name":"test_mse_dct_sparse_max_n","value":primary["test_mse_mean"],"SE":primary["test_mse_se"],"params":64,"nivel_rigor":c.rigor_level,"etiqueta":"SEÑAL"}; LEDGER.parent.mkdir(parents=True,exist_ok=True)
        with LEDGER.open("a",encoding="utf-8") as handle: handle.write(json.dumps(entry,ensure_ascii=False)+"\n")
        log.obj("LEDGER_ENTRY",entry); log.line("EXPERIMENT_COMPLETE | status=success | next_step=inspeccionar_JSON_antes_de_redactar_findings"); return 0
    except Exception as e: log.line(f"EXPERIMENT_ERROR | type={type(e).__name__} | message={e}"); return 1
if __name__=="__main__": raise SystemExit(main())
