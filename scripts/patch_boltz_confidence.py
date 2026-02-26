#!/usr/bin/env python3
"""
Patch Boltz2 confidencev2.py to fix pair_chains_iptm fallback for multi-chain complexes.

Bug: When compute_ptms() fails (common with 3+ chain complexes), the except handler
sets pair_chains_iptm to a flat tensor instead of the expected nested dict structure.
This crashes both the multiplicity aggregation loop and the output writer.

Fix: Build a proper nested dict {chain_id: {chain_id: zeros_tensor}} in the fallback.

Usage:
    python scripts/patch_boltz_confidence.py

Finds the installed Boltz package automatically and patches in-place (with backup).
"""

import importlib
import shutil
import sys
from pathlib import Path


def find_confidencev2():
    """Find the installed confidencev2.py file."""
    try:
        import boltz.model.modules.confidencev2 as mod
        return Path(mod.__file__)
    except ImportError:
        # Try common conda paths
        candidates = list(Path(sys.prefix).rglob("boltz/model/modules/confidencev2.py"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError("Cannot find boltz confidencev2.py")


# The buggy fallback (what we're replacing)
OLD_BLOCK = '''            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)'''

# The fixed fallback (builds proper nested dict)
NEW_BLOCK = '''            asym_ids_list = torch.unique(feats["asym_id"]).tolist()
            pair_chains_iptm = {}
            for _idx1 in asym_ids_list:
                chain_iptm = {}
                for _idx2 in asym_ids_list:
                    chain_iptm[_idx2] = torch.zeros_like(complex_plddt)
                pair_chains_iptm[_idx1] = chain_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm'''


def main():
    target = find_confidencev2()
    print(f"Found: {target}")

    content = target.read_text()

    if OLD_BLOCK not in content:
        if "pair_chains_iptm = {}" in content:
            print("Already patched!")
            return
        # Try to find it with different whitespace
        print("ERROR: Could not find the exact fallback block to patch.")
        print("The file may have been modified or is a different version.")
        print("Look for 'pair_chains_iptm' in the except block manually:")
        print(f"  {target}")
        sys.exit(1)

    # Backup
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")

    # Patch
    new_content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)

    # Verify only one replacement was made
    if new_content == content:
        print("ERROR: Replacement had no effect")
        sys.exit(1)

    target.write_text(new_content)
    print("Patched successfully!")
    print(f"  Fixed: pair_chains_iptm fallback now returns nested dict structure")
    print(f"  File:  {target}")


if __name__ == "__main__":
    main()
