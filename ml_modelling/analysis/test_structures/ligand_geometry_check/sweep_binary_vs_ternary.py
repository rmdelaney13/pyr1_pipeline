import numpy as np, os, glob
def parse(pdb):
    at={}; bonds={}
    for l in open(pdb):
        if l.startswith("HETATM") and l[21]=="B":
            s=int(l[6:11]); at[s]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        elif l.startswith("CONECT"):
            nums=[int(l[i:i+5]) for i in range(6,len(l.rstrip()),5) if l[i:i+5].strip()]
            if nums:
                bonds.setdefault(nums[0],set()).update(nums[1:])
                for b in nums[1:]: bonds.setdefault(b,set()).add(nums[0])
    return at,{a:set(x for x in bonds.get(a,()) if x in at) for a in at}
def rings(at,adj):
    R=set()
    for s0 in at:
        st=[(s0,[s0])]
        while st:
            c,p=st.pop()
            for nb in adj[c]:
                if nb==s0 and len(p) in (5,6): R.add(frozenset(p))
                elif nb not in p and len(p)<6: st.append((nb,p+[nb]))
    return [set(r) for r in R]
def maxdev(pdb):
    try: at,adj=parse(pdb)
    except: return None
    Rs=rings(at,adj)
    if not Rs: return None
    core=set().union(*Rs)
    if len(core)<15: return None
    c=np.array([at[s] for s in core]); C=c-c.mean(0); _,_,V=np.linalg.svd(C)
    return float(np.abs(C@V[2]).max())
BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
TER="ml_modelling/data/20260818_sort_data/output_ternary"
FLAT=0.7
def sweep(root,lig,k=60):
    v=[]
    for d in sorted(glob.glob(f"{root}/boltz_results_*__{lig}"))[:k]:
        n=d.split("boltz_results_")[1]
        m=maxdev(f"{d}/predictions/{n}/{n}_model_0.pdb")
        if m is not None: v.append(m)
    return np.array(v)
print(f"{'ligand':6s} {'complex':8s} {'n':>3s} {'median':>7s} {'q25':>5s} {'q75':>5s} {'%flat<0.7':>9s}")
summary={}
for lig in ["LCA","LCA3S","GLCA","CDCA"]:
    for cx,root in (("binary",BIN),("ternary",TER)):
        v=sweep(root,lig)
        pf=(v<FLAT).mean()*100
        summary[(lig,cx)]=(np.median(v),pf,len(v))
        print(f"{lig:6s} {cx:8s} {len(v):3d} {np.median(v):7.2f} {np.percentile(v,25):5.2f} {np.percentile(v,75):5.2f} {pf:8.0f}%")
    print()
print("=== binary -> ternary change in %flattened ===")
for lig in ["LCA","LCA3S","GLCA","CDCA"]:
    b=summary[(lig,'binary')][1]; t=summary[(lig,'ternary')][1]
    print(f"  {lig:6s} binary {b:3.0f}%  ternary {t:3.0f}%   (Δ {t-b:+.0f} pts)")
