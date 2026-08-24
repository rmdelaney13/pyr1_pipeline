import numpy as np, os

def parse(pdb):
    at={}; bonds={}
    for l in open(pdb):
        if l.startswith("HETATM") and l[21]=="B":
            s=int(l[6:11]); nm=l[12:16].strip(); el=l[76:78].strip()
            xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
            at[s]=(nm,el,xyz)
        elif l.startswith("CONECT"):
            nums=[int(l[i:i+5]) for i in range(6,len(l.rstrip()),5) if l[i:i+5].strip()]
            if nums:
                bonds.setdefault(nums[0],set()).update(nums[1:])
                for b in nums[1:]: bonds.setdefault(b,set()).add(nums[0])
    adj={a:set(x for x in bonds.get(a,()) if x in at) for a in at}
    return at,adj

def rings(at,adj,sizes=(5,6)):
    R=set()
    for start in at:
        stack=[(start,[start])]
        while stack:
            cur,path=stack.pop()
            for nb in adj[cur]:
                if nb==start and len(path) in sizes: R.add(frozenset(path))
                elif nb not in path and len(path)<max(sizes): stack.append((nb,path+[nb]))
    out={}
    for r in R:
        r=list(r); o=[r[0]]; prev=None; cur=r[0]; ok=True
        while len(o)<len(r):
            nx=[x for x in adj[cur] if x in r and x!=prev and x not in o]
            if not nx: ok=False; break
            prev,cur=cur,nx[0]; o.append(cur)
        if ok: out[frozenset(o)]=o
    return list(out.values())

def plane_dev(coords):
    C=coords-coords.mean(0)
    _,_,V=np.linalg.svd(C)
    z=C@V[2]
    return np.sqrt((z**2).mean()), np.abs(z).max()   # rms thickness, max half-thickness

def cp_Q(coords):
    Rr=coords-coords.mean(0); N=len(Rr); n=np.arange(N)
    Rp=(Rr*np.sin(2*np.pi*n/N)[:,None]).sum(0); Rpp=(Rr*np.cos(2*np.pi*n/N)[:,None]).sum(0)
    nv=np.cross(Rp,Rpp); nv/=np.linalg.norm(nv); z=Rr@nv
    return np.sqrt((z**2).sum())

targets={"LCA":"FDLMVVILYSKILGFM__LCA","LCA3S":"VLLAVGVVLWQILGFM__LCA3S",
         "GLCA":"IDLMVAVVLTKILGVI__GLCA","CDCA":"IDLMLAVVLTKIMGII__CDCA"}
BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
TER="ml_modelling/data/20260818_sort_data/output_ternary"

print(f"{'lig':6s}{'cplx':8s} {'6ring_Q':20s} {'5ring_Q':8s} {'core_thick_rms':15s} {'core_maxdev':11s} {'methyl_offplane':15s}")
for lig,n in targets.items():
    for cplx,root in (("binary",BIN),("ternary",TER)):
        p=f"{root}/boltz_results_{n}/predictions/{n}/{n}_model_0.pdb"
        at,adj=parse(p); Rs=rings(at,adj)
        six=[o for o in Rs if len(o)==6]; five=[o for o in Rs if len(o)==5]
        Q6=[cp_Q(np.array([at[s][2] for s in o])) for o in six]
        Q5=[cp_Q(np.array([at[s][2] for s in o])) for o in five]
        # core = union of all ring atoms
        core=set().union(*[set(o) for o in Rs]) if Rs else set()
        ccoords=np.array([at[s][2] for s in core])
        rms,mx=plane_dev(ccoords)
        # angular methyls: carbons bonded to a ring-fusion atom (deg3 in ring) and NOT in any ring, terminal (deg1)
        ringatoms=core
        methyls=[a for a in at if at[a][1]=="C" and a not in ringatoms and len(adj[a])==1 and any(nb in ringatoms for nb in adj[a])]
        # project methyl carbons onto core plane, measure off-plane distance
        C=ccoords-ccoords.mean(0); _,_,V=np.linalg.svd(C); nrm=V[2]; cen=ccoords.mean(0)
        moff=[abs((at[a][2]-cen)@nrm) for a in methyls]
        moff_s=" ".join(f"{x:.2f}" for x in sorted(moff))
        print(f"{lig:6s}{cplx:8s} {' '.join(f'{q:.2f}' for q in sorted(Q6)):20s} "
              f"{(f'{Q5[0]:.2f}' if Q5 else '-'):8s} {rms:<15.2f} {mx:<11.2f} {moff_s:15s}")
    print()
