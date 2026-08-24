import json, csv, shutil, os

BIN="/scratch/alpine/ryde3462/boltz_sort_20260818/output_holo"
TER="ml_modelling/data/20260818_sort_data/output_ternary"
META="ml_modelling/data/20260818_sort_data/structural_modeling_all_measured_pairs.csv"
OUT="ml_modelling/analysis/test_structures"

SEL={
 "LCA":  "FDLMVVILYSKILGFM__LCA",
 "LCA3S":"FDLMVVILYSKILGFM__LCA3S",
 "GLCA": "IDLMVAVVLTKILGVI__GLCA",
 "CDCA": "IDLMLAVVLTKIMGII__CDCA",
}

meta={r["structural_job_id"]:r for r in csv.DictReader(open(META))}

def conf(root,n):
    p=f"{root}/boltz_results_{n}/predictions/{n}/confidence_{n}_model_0.json"
    return json.load(open(p))
def aff(root,n):
    p=f"{root}/boltz_results_{n}/predictions/{n}/affinity_{n}.json"
    return json.load(open(p)) if os.path.exists(p) else {}
def pdb(root,n):
    return f"{root}/boltz_results_{n}/predictions/{n}/{n}_model_0.pdb"

os.makedirs(OUT,exist_ok=True)
rows=[]
for lig,n in SEL.items():
    pocket=n.split("__")[0]
    d=f"{OUT}/{lig}__{pocket}"
    os.makedirs(d,exist_ok=True)
    shutil.copy(pdb(BIN,n), f"{d}/binary.pdb")
    shutil.copy(pdb(TER,n), f"{d}/ternary.pdb")
    bc,tc=conf(BIN,n),conf(TER,n)
    ba,ta=aff(BIN,n),aff(TER,n)
    m=meta.get(n,{})
    def g(x,k):
        try:return round(float(x[k]),4)
        except:return ""
    rows.append(dict(
        ligand=lig, pocket_seq=pocket, structural_job_id=n,
        # NGS
        ER_mean=g(m,"ER_mean"), fold_enrichment=g(m,"fold_enrichment"),
        primary_q=m.get("primary_q",""), structural_roles=m.get("structural_roles",""),
        constitutive_high_confidence=m.get("constitutive_high_confidence",""),
        # binary boltz
        bin_confidence=g(bc,"confidence_score"), bin_ptm=g(bc,"ptm"),
        bin_iptm=g(bc,"iptm"), bin_ligand_iptm=g(bc,"ligand_iptm"),
        bin_protein_iptm=g(bc,"protein_iptm"), bin_complex_plddt=g(bc,"complex_plddt"),
        bin_complex_iplddt=g(bc,"complex_iplddt"),
        bin_affinity_pred=g(ba,"affinity_pred_value"), bin_affinity_prob=g(ba,"affinity_probability_binary"),
        # ternary boltz
        ter_confidence=g(tc,"confidence_score"), ter_ptm=g(tc,"ptm"),
        ter_iptm=g(tc,"iptm"), ter_ligand_iptm=g(tc,"ligand_iptm"),
        ter_protein_iptm=g(tc,"protein_iptm"), ter_complex_plddt=g(tc,"complex_plddt"),
        ter_complex_iplddt=g(tc,"complex_iplddt"),
        ter_affinity_pred=g(ta,"affinity_pred_value"), ter_affinity_prob=g(ta,"affinity_probability_binary"),
    ))

cols=list(rows[0].keys())
with open(f"{OUT}/test_structures_stats.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(rows)

# print summary
for r in rows:
    print(f"{r['ligand']:5s} {r['pocket_seq']}  ER={r['ER_mean']} fold={r['fold_enrichment']}  "
          f"bin[conf={r['bin_confidence']} ligiptm={r['bin_ligand_iptm']} aff_p={r['bin_affinity_prob']}]  "
          f"ter[conf={r['ter_confidence']} ligiptm={r['ter_ligand_iptm']}]")
print("\nWrote", f"{OUT}/test_structures_stats.csv")
