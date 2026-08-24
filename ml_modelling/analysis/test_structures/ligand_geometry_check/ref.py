import numpy as np, glob, itertools
def parse_pdb_dist(pdb):
    at={}
    for l in open(pdb):
        if l.startswith(("HETATM","ATOM")):
            el=l[76:78].strip() or l[12:16].strip()[0]
            if el=="H": continue
            s=int(l[6:11]); at[s]=(el,np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]))
    adj={a:set() for a in at}
    ks=list(at)
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            if np.linalg.norm(at[ks[i]][1]-at[ks[j]][1])<1.8:
                adj[ks[i]].add(ks[j]); adj[ks[j]].add(ks[i])
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
def normal(c): C=c-c.mean(0); _,_,V=np.linalg.svd(C); return V[2]
def thick(c): C=c-c.mean(0); _,_,V=np.linalg.svd(C); z=C@V[2]; return np.sqrt((z**2).mean()),np.abs(z).max()
print("REFERENCE RDKit LCA conformers (campaigns/LCA/conformers/conformers_final):")
for p in sorted(glob.glob("campaigns/LCA/conformers/conformers_final/conf_00*.pdb"))[:6]:
    at,adj=parse_pdb_dist(p); Rs=rings(at,adj)
    six=[r for r in Rs if len(r)==6]
    if len(six)<3: 
        print(f"  {p.split('/')[-1]}: only {len(six)} six-rings found"); continue
    core=set().union(*Rs); rms,mx=thick(np.array([at[s][1] for s in core]))
    norms=[normal(np.array([at[s][1] for s in r])) for r in six]
    ang=sorted(np.degrees(np.arccos(min(1,abs(np.dot(norms[i],norms[j]))))) for i,j in itertools.combinations(range(len(norms)),2))
    print(f"  {p.split('/')[-1]}: ncore={len(core)} thick_rms={rms:.2f} maxdev={mx:.2f}  ring-ring bend={' '.join(f'{a:.0f}' for a in ang)}")
