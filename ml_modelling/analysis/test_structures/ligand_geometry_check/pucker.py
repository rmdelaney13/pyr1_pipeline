import numpy as np, itertools, glob, os

def parse(pdb):
    at={}   # serial -> (name,elem,xyz), chain B only
    bonds={}
    for l in open(pdb):
        if l.startswith("HETATM") and l[21]=="B":
            s=int(l[6:11]); nm=l[12:16].strip(); el=l[76:78].strip()
            xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
            at[s]=(nm,el,xyz)
        elif l.startswith("CONECT"):
            nums=[int(l[i:i+5]) for i in range(6,len(l.rstrip()),5) if l[i:i+5].strip()]
            if nums:
                a=nums[0]; bonds.setdefault(a,set()).update(nums[1:])
                for b in nums[1:]: bonds.setdefault(b,set()).add(a)
    # restrict bonds to ligand atoms
    adj={a:set(x for x in bonds.get(a,()) if x in at) for a in at}
    return at,adj

def find_rings(at,adj,sizes=(5,6)):
    rings=set()
    nodes=list(at)
    for start in nodes:
        # DFS paths returning to start
        stack=[(start,[start])]
        while stack:
            cur,path=stack.pop()
            for nb in adj[cur]:
                if nb==start and len(path)>=min(sizes):
                    if len(path) in sizes:
                        rings.add(frozenset(path))
                elif nb not in path and len(path)<max(sizes):
                    stack.append((nb,path+[nb]))
    # order atoms cyclically for each ring
    ordered=[]
    for r in rings:
        r=list(r); # build order by walking adjacency
        o=[r[0]]; prev=None; cur=r[0]
        ok=True
        while len(o)<len(r):
            nxts=[x for x in adj[cur] if x in r and x!=prev and x not in o]
            if not nxts: ok=False; break
            prev=cur; cur=nxts[0]; o.append(cur)
        if ok and cur in r: ordered.append(o)
    # dedupe
    uniq={}; 
    for o in ordered: uniq[frozenset(o)]=o
    return list(uniq.values())

def cremer_pople_Q(coords):
    # coords: N x 3 in ring order
    R=coords-coords.mean(0)
    N=len(R)
    n=np.arange(N)
    Rp=(R*np.sin(2*np.pi*n/N)[:,None]).sum(0)
    Rpp=(R*np.cos(2*np.pi*n/N)[:,None]).sum(0)
    nvec=np.cross(Rp,Rpp); nvec/=np.linalg.norm(nvec)
    z=R@nvec
    Q=np.sqrt((z**2).sum())
    rms=np.sqrt((z**2).mean())
    return Q,rms

targets={
 "LCA":  "FDLMVVILYSKILGFM__LCA",
 "LCA3S":"VLLAVGVVLWQILGFM__LCA3S",
 "GLCA": "IDLMVAVVLTKILGVI__GLCA",
 "CDCA": "IDLMLAVVLTKIMGII__CDCA",
}
BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
TER="ml_modelling/data/20260818_sort_data/output_ternary"

print(f"{'lig':6s}{'cplx':8s}{'model':6s} {'nC-rings':9s} {'Q_per_6ring (Å)':30s}")
for lig,n in targets.items():
    for cplx,root in (("binary",BIN),("ternary",TER)):
        for m in range(5):
            p=f"{root}/boltz_results_{n}/predictions/{n}/{n}_model_{m}.pdb"
            if not os.path.exists(p): continue
            at,adj=parse(p)
            rings=find_rings(at,adj)
            # carbon-only 6-membered rings (steroid A/B/C); report all ring Q
            Qs=[]
            for o in rings:
                if len(o)==6:
                    coords=np.array([at[s][2] for s in o])
                    Q,rms=cremer_pople_Q(coords)
                    Qs.append(Q)
            Qs=sorted(Qs)
            qstr=" ".join(f"{q:.2f}" for q in Qs)
            print(f"{lig:6s}{cplx:8s}m{m:<5d} {len([r for r in rings if len(r)==6]):<9d} {qstr}")
    print()
