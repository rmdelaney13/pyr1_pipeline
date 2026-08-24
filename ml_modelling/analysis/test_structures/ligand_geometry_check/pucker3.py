import numpy as np, os
def parse(pdb):
    at={}; bonds={}
    for l in open(pdb):
        if l.startswith("HETATM") and l[21]=="B":
            s=int(l[6:11]); nm=l[12:16].strip(); el=l[76:78].strip()
            at[s]=(nm,el,np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]))
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
def normal(coords):
    C=coords-coords.mean(0); _,_,V=np.linalg.svd(C); return V[2],coords.mean(0)
def thick(coords):
    C=coords-coords.mean(0); _,_,V=np.linalg.svd(C); z=C@V[2]; return np.sqrt((z**2).mean()),np.abs(z).max()

targets={"LCA":"FDLMVVILYSKILGFM__LCA","LCA3S":"VLLAVGVVLWQILGFM__LCA3S",
         "GLCA":"IDLMVAVVLTKILGVI__GLCA","CDCA":"IDLMLAVVLTKIMGII__CDCA"}
BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
for lig,n in targets.items():
    print(f"=== {lig} (binary, all 5 models) ===")
    for m in range(5):
        p=f"{BIN}/boltz_results_{n}/predictions/{n}/{n}_model_{m}.pdb"
        at,adj=parse(p); Rs=rings(at,adj)
        six=[r for r in Rs if len(r)==6]
        core=set().union(*Rs)
        rms,mx=thick(np.array([at[s][2] for s in core]))
        # bend: max angle between any two 6-ring normals (deg)
        norms=[normal(np.array([at[s][2] for s in r]))[0] for r in six]
        ang=[]
        for i in range(len(norms)):
            for j in range(i+1,len(norms)):
                c=abs(np.dot(norms[i],norms[j])); ang.append(np.degrees(np.arccos(min(1,c))))
        print(f"  m{m}: ncore={len(core):2d} thick_rms={rms:.2f} maxdev={mx:.2f}  ring-ring bend(deg)={' '.join(f'{a:.0f}' for a in sorted(ang))}")
    print()
