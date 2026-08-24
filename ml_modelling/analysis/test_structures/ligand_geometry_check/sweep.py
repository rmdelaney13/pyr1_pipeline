import numpy as np, os, glob
def parse(pdb):
    at={}; bonds={}
    for l in open(pdb):
        if l.startswith("HETATM") and l[21]=="B":
            s=int(l[6:11]); el=l[76:78].strip()
            at[s]=(el,np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]))
        elif l.startswith("CONECT"):
            nums=[int(l[i:i+5]) for i in range(6,len(l.rstrip()),5) if l[i:i+5].strip()]
            if nums:
                bonds.setdefault(nums[0],set()).update(nums[1:])
                for b in nums[1:]: bonds.setdefault(b,set()).add(nums[0])
    adj={a:set(x for x in bonds.get(a,()) if x in at) for a in at}
    return at,adj
def rings(at,adj):
    R=set()
    for start in at:
        st=[(start,[start])]
        while st:
            cur,path=st.pop()
            for nb in adj[cur]:
                if nb==start and len(path) in (5,6): R.add(frozenset(path))
                elif nb not in path and len(path)<6: st.append((nb,path+[nb]))
    return [set(r) for r in R]
def thick(c): C=c-c.mean(0); _,_,V=np.linalg.svd(C); z=C@V[2]; return np.abs(z).max()
BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
def sample(lig,k=25):
    dirs=sorted(glob.glob(f"{BIN}/boltz_results_*__{lig}"))[:k]
    vals=[]
    for d in dirs:
        n=d.split("boltz_results_")[1]
        p=f"{d}/predictions/{n}/{n}_model_0.pdb"
        if not os.path.exists(p): continue
        at,adj=parse(p); Rs=rings(at,adj)
        if not Rs: continue
        core=set().union(*Rs)
        if len(core)<15: continue
        vals.append(thick(np.array([at[s][1] for s in core])))
    return np.array(vals)
for lig in ["LCA","LCA3S","GLCA","CDCA"]:
    v=sample(lig)
    flat=(v<0.7).mean()*100
    print(f"{lig:6s} n={len(v):2d}  core_maxdev: median={np.median(v):.2f} min={v.min():.2f} max={v.max():.2f}  | %flattened(<0.7Å)={flat:.0f}%")
