"""v333: curva rate-distortion de matrices U^T C V con K coeficientes."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"results"/"raw"; LEDGER=ROOT/"results"/"master_ledger.jsonl"
D=64; KS=(16,64,256,1024,4096); RIDGES=(1e-5,1e-3,1e-1); TEACHERS=("dct2d_decay","random2d_decay","dense_unstructured"); STUDENTS=("dense_linear","dct2d_topk","random2d_topk")
@dataclass(frozen=True)
class Cfg: experiment_id:str; mode:str; rigor_level:int; seeds:tuple[int,...]; ns:tuple[int,...]; test_n:int; val_n:int; noise:float=0.05
class Log:
 def __init__(self): self.t=time.perf_counter()
 def line(self,s):
  e=time.perf_counter()-self.t;h,r=divmod(e,3600);m,x=divmod(r,60)
  for q in str(s).splitlines() or [""]:print(f"[+{int(h):02d}:{int(m):02d}:{x:05.2f}] {q}",flush=True)
 def obj(self,k,x):self.line(k+":");self.line(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,default=str))
def dct():
 i=torch.arange(D,dtype=torch.float64)[:,None];j=torch.arange(D,dtype=torch.float64)[None,:];u=torch.cos(math.pi/D*(i+.5)*j);u[:,0]/=math.sqrt(D);u[:,1:]*=math.sqrt(2/D);return u
def qmat(seed):
 g=torch.Generator().manual_seed(seed);q,r=torch.linalg.qr(torch.randn(D,D,generator=g,dtype=torch.float64));return q*torch.sign(torch.diag(r))[None,:]
def sha(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
ORDER=sorted(((i,j) for i in range(D) for j in range(D)),key=lambda ij:(ij[0]+ij[1],ij[0],ij[1]))
def mask(k):
 out=torch.zeros(D,D,dtype=torch.bool)
 for i,j in ORDER[:k]:out[i,j]=True
 return out
def bases(student,seed):
 if student=="dense_linear":return None,None,{"basis":"none"}
 b=dct() if student=="dct2d_topk" else qmat(333000+seed);return b,b,{"basis":"dct2" if student=="dct2d_topk" else "random_q","sha256":sha(b),"seed":None if student=="dct2d_topk" else 333000+seed}
def teacher(name,seed):
 if name=="dense_unstructured":
  g=torch.Generator().manual_seed(334000+seed);w=torch.randn(D,D,generator=g,dtype=torch.float64)/math.sqrt(D);return w,{"teacher":name,"sha256":sha(w)}
 b=dct() if name=="dct2d_decay" else qmat(333000+seed);c=torch.zeros(D,D,dtype=torch.float64)
 for rank,(i,j) in enumerate(ORDER):c[i,j]=1/(1+rank)**.7
 w=b.T@c@b;return w,{"teacher":name,"basis_sha256":sha(b),"matrix_sha256":sha(w)}
def fit_dense(x,y,r):return torch.linalg.solve(x.T@x+r*torch.eye(D,dtype=torch.float64),x.T@y)
def fit_spec(x,y,u,v,m,r):
 z=x@u.T;t=y@v.T;c=torch.zeros(D,D,dtype=torch.float64)
 for j in range(D):
  ix=torch.where(m[:,j])[0]
  if len(ix): c[ix,j]=torch.linalg.solve(z[:,ix].T@z[:,ix]+r*torch.eye(len(ix),dtype=torch.float64),z[:,ix].T@t[:,j])
 return c
def pred(student,p,x,u,v):return x@p if student=="dense_linear" else (x@u.T)@p@v
def mse(a,b):return (a-b).square().mean().item()
def quant(p,bits):
 lo,hi=p.min(),p.max();levels=2**bits-1
 return p if (hi-lo).abs()<1e-14 else torch.round((p-lo)/(hi-lo)*levels)/levels*(hi-lo)+lo
def meanse(xs):
 t=torch.tensor(xs,dtype=torch.float64);sd=0 if len(t)==1 else t.std(unbiased=True).item();return {"mean":t.mean().item(),"sd":sd,"se":sd/math.sqrt(len(t))}
def cfg(a):
 lvl,seeds,ns,test,val=(1,(42,),(16,64,256),2048,256) if a.mode=="pilot" else (2,(10,20,30,42,100),(16,32,64,128,256),8192,512)
 if a.seeds:seeds=tuple(map(int,a.seeds.split(",")))
 return Cfg(a.run_id,a.mode,lvl,seeds,ns,test,val)
def main():
 log=Log()
 try:
  p=argparse.ArgumentParser();p.add_argument("--mode",choices=("pilot","level2"),default="pilot");p.add_argument("--run-id",default="v333_rate_distortion_spectral_matrix");p.add_argument("--seeds",default=None);a=p.parse_args();c=cfg(a);start=datetime.now(timezone.utc).isoformat()
  log.line("="*96);log.line(f"EXPERIMENT_START | {c.experiment_id} — rate-distortion spectral matrix");log.line("QUESTION | ¿Cómo cambia MSE/bits/coste al aumentar K hasta d²?");log.line("CLAIM_BOUNDARY | matriz sintética; capacidad total recuperada sólo en K=d².");log.obj("FULL_CONFIG",asdict(c));log.line("ARCHITECTURE | Dense=4096 coeficientes; espectral=K coeficientes C en W=U^T C V; ridge validado.")
  rows=[]
  for seed in c.seeds:
   for tn in TEACHERS:
    w,tm=teacher(tn,seed);g=torch.Generator().manual_seed(335000+seed);xt=torch.randn(c.test_n,D,generator=g,dtype=torch.float64);yt=xt@w+c.noise*torch.randn(c.test_n,D,generator=g,dtype=torch.float64);xv=torch.randn(c.val_n,D,generator=g,dtype=torch.float64);yv=xv@w+c.noise*torch.randn(c.val_n,D,generator=g,dtype=torch.float64);log.obj("TEACHER",{"seed":seed,**tm})
    for n in c.ns:
     gx=torch.Generator().manual_seed(336000+seed*100+n+TEACHERS.index(tn));x=torch.randn(n,D,generator=gx,dtype=torch.float64);y=x@w+c.noise*torch.randn(n,D,generator=gx,dtype=torch.float64)
     for st in STUDENTS:
      u,v,bm=bases(st,seed)
      for k in KS:
       m=mask(k);best=None
       for r in RIDGES:
        pfit=fit_dense(x,y,r) if st=="dense_linear" else fit_spec(x,y,u,v,m,r);score=mse(pred(st,pfit,xv,u,v),yv)
        if best is None or score<best[0]:best=(score,r,pfit)
       _,r,pfit=best;t0=time.perf_counter();_ = pred(st,pfit,xt,u,v);forward=time.perf_counter()-t0;test=mse(pred(st,pfit,xt,u,v),yt);effective=4096 if st=="dense_linear" else k
       q={}
       for b in (4,8,16):
        qp=quant(pfit,b) if st=="dense_linear" else torch.zeros_like(pfit); qp=qp if st=="dense_linear" else qp.masked_scatter(m,quant(pfit[m],b));q[str(b)]={"bits":int(effective*b+32),"test_mse":mse(pred(st,qp,xt,u,v),yt)}
       row={"seed":seed,"teacher":tn,"n_train":n,"student":st,"K":4096 if st=="dense_linear" else k,"params":effective,"ridge":r,"basis":bm,"test_mse":test,"forward_seconds":forward,"quantization":q};rows.append(row);log.line(f"POINT | seed={seed} | teacher={tn} | n={n} | student={st} | K={row['K']} | ridge={r:.0e} | test_mse={test:.8f} | bits4={q['4']['bits']} | forward_seconds={forward:.6f}")
  summary={}
  for tn in TEACHERS:
   for n in c.ns:
    for st in STUDENTS:
     for k in KS:
      rs=[r for r in rows if r['teacher']==tn and r['n_train']==n and r['student']==st and r['K']==(4096 if st=='dense_linear' else k)]
      if rs:summary[f"{tn}|n={n}|{st}|K={k}"]={"teacher":tn,"n_train":n,"student":st,"K":k,"n_seeds":len(rs),"test_mse":meanse([r['test_mse'] for r in rs]),"forward_seconds":meanse([r['forward_seconds'] for r in rs])}
  log.obj("SUMMARY",summary);payload={"experiment_id":c.experiment_id,"started_utc":start,"config":asdict(c),"results":rows,"summary":summary};RAW.mkdir(parents=True,exist_ok=True);out=RAW/f"{c.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json";out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");log.line(f"ARTIFACT | raw_results={out}");log.line("EXPERIMENT_COMPLETE | status=success | next_step=inspeccionar_JSON_antes_de_redactar_findings");return 0
 except Exception as e:log.line(f"EXPERIMENT_ERROR | type={type(e).__name__} | message={e}");return 1
if __name__=="__main__":raise SystemExit(main())
