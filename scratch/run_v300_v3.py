import math, time, os, json, gc, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SCRIPT_START = time.perf_counter()
def log(m=""): print(f"[+{int(time.perf_counter()-SCRIPT_START)//3600:02d}:{int(time.perf_counter()-SCRIPT_START)%3600//60:02d}:{int(time.perf_counter()-SCRIPT_START)%60:02d}] {m}", flush=True)

CFG = {
    "d_k_list": [32, 64, 128],
    "iso_floats_map": {32: (32,45), 64: (64,90), 128: (128,181)},
    "num_pairs_list": [32, 64, 128, 256],
    "num_keys": 256, "num_vals": 256,
    "batch_size": 32, "n_layers": 3, "epochs": 15, "steps_per_epoch": 50,
    "lr_grid": [2e-3, 4e-3], "seed": 42, "chunk_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CFG["device"])
if device.type=="cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch._dynamo.config.cache_size_limit=64

PAD_ID=0; KEY_OFFSET=1; VAL_OFFSET=1+CFG["num_keys"]; QUERY_MARKER=VAL_OFFSET+CFG["num_vals"]; VOCAB_SIZE=QUERY_MARKER+1

def generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, device=device):
    rand_k=torch.rand(batch_size, CFG["num_keys"], device=device)
    keys=torch.argsort(rand_k,dim=-1)[:,:num_pairs]+KEY_OFFSET
    vals=torch.randint(0,CFG["num_vals"],(batch_size,num_pairs),device=device)+VAL_OFFSET
    x=torch.full((batch_size,seq_len),PAD_ID,dtype=torch.long,device=device)
    y=torch.full((batch_size,seq_len),-100,dtype=torch.long,device=device)
    x[:,:2*num_pairs]=torch.stack([keys,vals],dim=2).view(batch_size,2*num_pairs)
    query_perm=torch.argsort(torch.rand(batch_size,num_pairs,device=device),dim=-1)
    q_keys=torch.gather(keys,1,query_perm); q_vals=torch.gather(vals,1,query_perm)
    n_queries=min(num_pairs,(seq_len-2*num_pairs-2)//2)
    pos_q=(2*num_pairs+2+2*torch.arange(n_queries,device=device)).unsqueeze(0).expand(batch_size,-1)
    x.scatter_(1,pos_q,QUERY_MARKER); x.scatter_(1,pos_q+1,q_keys[:,:n_queries])
    y.scatter_(1,pos_q+1,q_vals[:,:n_queries])
    return x,y

# --- MODELO RAPIDO : Complex descompuesto en 2 reales -> compilable ---
class SinCosPE(nn.Module):
    def __init__(self,d_model,max_len=4096):
        super().__init__()
        pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.shape[1]]

class ShortCausalConv1D(nn.Module):
    def __init__(self,d_model,k=4):
        super().__init__(); self.conv=nn.Conv1d(d_model,d_model,k,padding=k-1,groups=d_model); self.act=nn.SiLU()
    def forward(self,x): return x+self.act(self.conv(x.transpose(1,2))[:,:,:x.shape[1]].transpose(1,2))

class FFN(nn.Module):
    def __init__(self,d_model,expand=2):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d_model,d_model*expand),nn.SiLU(),nn.Linear(d_model*expand,d_model))
    def forward(self,x): return self.net(x)

class ComplexDeltaPhaseFastBlock(nn.Module):
    def __init__(self,d_model,n_heads=2,d_k=64,chunk_size=64):
        super().__init__()
        self.d_model,self.n_heads,self.d_k,self.chunk_size=d_model,n_heads,d_k,chunk_size
        self.norm1,self.norm2=nn.LayerNorm(d_model),nn.LayerNorm(d_model)
        self.causal_conv=ShortCausalConv1D(d_model,4)
        self.theta_k_proj=nn.Linear(d_model,n_heads*d_k); self.theta_q_proj=nn.Linear(d_model,n_heads*d_k)
        self.val_proj=nn.Linear(d_model,n_heads*d_k); self.beta_proj=nn.Linear(d_model,n_heads)
        self.out_proj=nn.Linear(n_heads*d_k,d_model); self.ffn=FFN(d_model)

    def forward(self,x):
        res=x; conv_x=self.causal_conv(self.norm1(x))
        B,L,D=conv_x.shape
        theta_k=self.theta_k_proj(conv_x).view(B,L,self.n_heads,self.d_k)
        theta_q=self.theta_q_proj(conv_x).view(B,L,self.n_heads,self.d_k)
        v=self.val_proj(conv_x).view(B,L,self.n_heads,self.d_k)
        beta=torch.sigmoid(self.beta_proj(conv_x)).view(B,L,self.n_heads,1,1)
        # precalcula cos/sin una vez
        cos_k=torch.cos(theta_k); sin_k=torch.sin(theta_k)
        cos_q=torch.cos(theta_q); sin_q=torch.sin(theta_q)
        
        # Estado recurrente en reales: M = Mr + i*Mi
        Mr=torch.zeros(B,self.n_heads,self.d_k,self.d_k,device=x.device,dtype=x.dtype)
        Mi=torch.zeros_like(Mr)
        inv_dk=1.0/float(self.d_k)
        outputs=[]
        # Chunked scan con checkpoint -> O(chunk) memoria
        for s in range(0,L,self.chunk_size):
            e=min(s+self.chunk_size,L)
            # funcion pura para checkpoint
            def chunk_fn(Mr_c, Mi_c, cos_k_c, sin_k_c, cos_q_c, sin_q_c, v_c, beta_c):
                # loop dentro del chunk sigue siendo python pero el grafo se libera entre chunks
                out_chunk=[]
                for t in range(cos_k_c.shape[1]):
                    ck=cos_k_c[:,t]; sk=sin_k_c[:,t]; cq=cos_q_c[:,t]; sq=sin_q_c[:,t]
                    vt=v_c[:,t]; bt=beta_c[:,t]
                    # v_old = Re(M @ k*) / dk = (Mr@ck + Mi@sk)/dk
                    # usamos einsum que compila a bmm
                    v_old=(torch.einsum('b h i j, b h j -> b h i', Mr_c, ck) + torch.einsum('b h i j, b h j -> b h i', Mi_c, sk))*inv_dk
                    err=vt - v_old
                    # M += beta * err outer k
                    # unsqueeze para outer: err (B,H,dk) , ck (B,H,dk) -> (B,H,dk,dk)
                    if bt.shape[-1]==1: bt=bt.squeeze(-1).squeeze(-1) # B,H
                    bt_=bt.unsqueeze(-1).unsqueeze(-1) # B,H,1,1
                    Mr_c = Mr_c + bt_ * torch.einsum('b h i, b h j -> b h i j', err, ck)
                    Mi_c = Mi_c + bt_ * torch.einsum('b h i, b h j -> b h i j', err, sk)
                    # ret = Re(M_new @ q*)/dk
                    ret=(torch.einsum('b h i j, b h j -> b h i', Mr_c, cq) + torch.einsum('b h i j, b h j -> b h i', Mi_c, sq))*inv_dk
                    out_chunk.append(ret)
                return Mr_c, Mi_c, torch.stack(out_chunk,dim=1)
            
            # checkpoint solo en train
            if self.training:
                Mr, Mi, out_c = checkpoint(chunk_fn, Mr, Mi, cos_k[:,s:e], sin_k[:,s:e], cos_q[:,s:e], sin_q[:,s:e], v[:,s:e], beta[:,s:e], use_reentrant=False)
            else:
                Mr, Mi, out_c = chunk_fn(Mr, Mi, cos_k[:,s:e], sin_k[:,s:e], cos_q[:,s:e], sin_q[:,s:e], v[:,s:e], beta[:,s:e])
            outputs.append(out_c)
        retrieved=torch.cat(outputs,dim=1).view(B,L,self.n_heads*self.d_k)
        attn_out=self.out_proj(retrieved)
        return res + attn_out + self.ffn(self.norm2(res+attn_out))

# Real y MHA quedan igual pero ahora tambien se compilan
class RealDeltaNetVanillaBlock(nn.Module):
    def __init__(self,d_model,n_heads=2,d_k_real=90,chunk_size=64):
        super().__init__(); self.d_model,self.n_heads,self.d_k=d_model,n_heads,d_k_real; self.chunk_size=chunk_size
        self.norm1,self.norm2=nn.LayerNorm(d_model),nn.LayerNorm(d_model); self.causal_conv=ShortCausalConv1D(d_model,4)
        self.k_proj=nn.Linear(d_model,n_heads*self.d_k); self.q_proj=nn.Linear(d_model,n_heads*self.d_k)
        self.val_proj=nn.Linear(d_model,n_heads*self.d_k); self.beta_proj=nn.Linear(d_model,n_heads)
        self.out_proj=nn.Linear(n_heads*self.d_k,d_model); self.ffn=FFN(d_model)
    def forward(self,x):
        res=x; conv_x=self.causal_conv(self.norm1(x)); B,L,D=conv_x.shape
        K=F.normalize(self.k_proj(conv_x).view(B,L,self.n_heads,self.d_k),p=2,dim=-1)
        Q=F.normalize(self.q_proj(conv_x).view(B,L,self.n_heads,self.d_k),p=2,dim=-1)
        v=self.val_proj(conv_x).view(B,L,self.n_heads,self.d_k); beta=torch.sigmoid(self.beta_proj(conv_x)).view(B,L,self.n_heads,1,1)
        M=torch.zeros(B,self.n_heads,self.d_k,self.d_k,device=x.device,dtype=x.dtype); outs=[]
        for s in range(0,L,self.chunk_size):
            e=min(s+self.chunk_size,L)
            def chunk_fn(Mc,Kc,Qc,vc,bc):
                oc=[]
                for t in range(Kc.shape[1]):
                    k,q,vt,bt=Kc[:,t],Qc[:,t],vc[:,t],bc[:,t]
                    v_old=torch.einsum('b h i j, b h j -> b h i', Mc, k)
                    err=vt-v_old; Mc=Mc+bt*torch.einsum('b h i, b h j -> b h i j', err, k)
                    ret=torch.einsum('b h i j, b h j -> b h i', Mc, q); oc.append(ret)
                return Mc, torch.stack(oc,dim=1)
            if self.training: M,out_c=checkpoint(chunk_fn,M,K[:,s:e],Q[:,s:e],v[:,s:e],beta[:,s:e],use_reentrant=False)
            else: M,out_c=chunk_fn(M,K[:,s:e],Q[:,s:e],v[:,s:e],beta[:,s:e])
            outs.append(out_c)
        retrieved=torch.cat(outs,dim=1).view(B,L,self.n_heads*self.d_k); attn_out=self.out_proj(retrieved)
        return res+attn_out+self.ffn(self.norm2(res+attn_out))

class CausalAttentionBlock(nn.Module):
    def __init__(self,d_model,n_heads=2):
        super().__init__(); self.causal_conv=ShortCausalConv1D(d_model,4)
        self.mha=nn.MultiheadAttention(d_model,n_heads,batch_first=True)
        self.norm1,self.norm2=nn.LayerNorm(d_model),nn.LayerNorm(d_model); self.ffn=FFN(d_model)
        max_len=max(4*p+2 for p in CFG["num_pairs_list"]); self.register_buffer("causal_mask",torch.triu(torch.ones(max_len,max_len,dtype=torch.bool),1))
    def forward(self,x):
        res=x; conv_x=self.causal_conv(self.norm1(x)); B,L,D=conv_x.shape
        a,_=self.mha(conv_x,conv_x,conv_x,attn_mask=self.causal_mask[:L,:L],is_causal=False)
        return res+a+self.ffn(self.norm2(res+a))

class SequenceModel(nn.Module):
    def __init__(self,block_cls,vocab_size,d_model,n_layers=3,block_kwargs=None):
        super().__init__(); self.emb=nn.Embedding(vocab_size,d_model); self.pe=SinCosPE(d_model)
        self.layers=nn.ModuleList([block_cls(d_model=d_model,**(block_kwargs or {})) for _ in range(n_layers)]); self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        h=self.pe(self.emb(x))
        for l in self.layers: h=l(h)
        return self.head(h)

def train_and_eval(model, name, num_pairs, seq_len, lr):
    # AHORA COMPILAMOS TODO, incluido el complejo real-descompuesto
    if device.type=="cuda":
        model=torch.compile(model, mode="max-autotune", fullgraph=False)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0)
    crit=nn.CrossEntropyLoss(ignore_index=-100)
    micro=2 if seq_len>=2048 else (4 if seq_len>=1024 else (16 if seq_len>=512 else CFG["batch_size"]))
    accum=max(1,CFG["batch_size"]//micro)
    model.train(); t0=time.perf_counter()
    for ep in range(CFG["epochs"]):
        for step in range(CFG["steps_per_epoch"]):
            opt.zero_grad(set_to_none=True)
            tot_loss=0
            for _ in range(accum):
                xb,yb=generate_mqar_batch_vectorized(micro,num_pairs,seq_len,device=device) # streaming!
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16, enabled=device.type=='cuda'):
                    logits=model(xb)
                    loss=crit(logits.view(-1,VOCAB_SIZE),yb.view(-1))/accum
                loss.backward(); tot_loss+=loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        log(f"  [{name:28s} lr={lr:.4f}] Epoch {ep+1}/{CFG['epochs']} loss={tot_loss.item():.4f}")
    # eval
    model.eval(); correct=total=0
    with torch.no_grad():
        for _ in range(10):
            xb,yb=generate_mqar_batch_vectorized(micro,num_pairs,seq_len,device=device)
            pred=model(xb).argmax(-1); mask=yb!=-100
            correct+=(pred[mask]==yb[mask]).sum().item(); total+=mask.sum().item()
    if device.type=="cuda": torch.cuda.synchronize()
    acc=correct/total*100 if total else 0
    del model; gc.collect(); torch.cuda.empty_cache()
    return acc, time.perf_counter()-t0