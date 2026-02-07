#!/usr/bin/env python3
import os
import argparse
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def extract_sequence_from_pdb(pdb_path, chain_id=None):
    """
    Parse the PDB file at pdb_path, extract the amino acid sequence for the specified chain.
    If chain_id is None, we take the first chain found.
    Returns a Bio.Seq.Seq object.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)

    # Choose first model (most PDBs have only one)
    model = next(structure.get_models())

    # If chain_id is provided, try to get that chain; otherwise, take the first chain
    if chain_id:
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())

    # Use PPBuilder to build polypeptides from ATOM records
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)

    if not polypeptides:
        raise ValueError(f"No polypeptide found in chain {chain.id} of {pdb_path}")

    # Concatenate all fragments (in case of breaks)
    sequence_str = "".join(str(pp.get_sequence()) for pp in polypeptides)
    return Seq(sequence_str)


def apply_mutations_and_insertion(seq):
    """
    Given a Bio.Seq.Seq object, apply the following point mutations (1-based indexing)
    and then insert "QN" at position 69 (1-based), shifting downstream residues:

        S29Q
        E43D
        D78E
        N88S
        T116R
        R132K

    After mutating, insert "QN" just before the residue that was originally at position 69
    (so that "VEFE" at positions 67-70 becomes "VEQNFE").
    Returns a new Bio.Seq.Seq with all edits applied.
    """

    # Convert to mutable Python list for easy indexing
    seq_list = list(str(seq))

    # Define the point mutations: {1-based_position: new_residue}
    mutations = {
        29: "Q",
        43: "D",
        78: "E",
        88: "S",
        116: "R",
        132: "K"
    }

    # Apply each point mutation (convert 1-based to 0-based index)
    for pos1, new_res in mutations.items():
        idx0 = pos1 - 1
        if idx0 < 0 or idx0 >= len(seq_list):
            raise IndexError(f"Mutation position {pos1} out of range for sequence length {len(seq_list)}")
        seq_list[idx0] = new_res

    # Now insert "QN" at position 69 (1-based), which is index 68 in 0-based.
    insert_idx0 = 69 - 1
    if insert_idx0 < 0 or insert_idx0 > len(seq_list):
        raise IndexError(f"Insertion index {insert_idx0} out of range for sequence length {len(seq_list)}")

    seq_list.insert(insert_idx0, "Q")
    seq_list.insert(insert_idx0 + 1, "N")

    # Join back into a string
    new_seq_str = "".join(seq_list)
    return Seq(new_seq_str)


def main(pdb_dir, output_fasta, chain_id=None):
    """
    Iterate over all .pdb files in pdb_dir, extract their chain sequence,
    apply mutations + insertion, remove duplicates, and write unique sequences
    into output_fasta in FASTA format.
    """
    records = []
    seen_sequences = set()  # to track unique sequence strings

    for fname in sorted(os.listdir(pdb_dir)):
        if not fname.lower().endswith(".pdb"):
            continue

        pdb_path = os.path.join(pdb_dir, fname)
        try:
            orig_seq = extract_sequence_from_pdb(pdb_path, chain_id=chain_id)
        except Exception as e:
            print(f"Warning: could not extract sequence from {fname}: {e}")
            continue

        mutated_seq = apply_mutations_and_insertion(orig_seq)
        seq_str = str(mutated_seq)

        # Skip if this sequence was already seen
        if seq_str in seen_sequences:
            continue

        seen_sequences.add(seq_str)
        record_id = os.path.splitext(fname)[0]
        record = SeqRecord(mutated_seq, id=record_id, description="")
        records.append(record)

    if not records:
        print("No valid PDB sequences were found or all extractions failed.")
        return

    # Write out unique sequences to the FASTA file
    with open(output_fasta, "w") as out_f:
        SeqIO.write(records, out_f, "fasta")

    print(f"Wrote {len(records)} unique sequences (with pocket mutations + insertion) to {output_fasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate sequences from all PDBs in a directory, apply pocket mutations + insertion, remove duplicates, and write to a FASTA file."
    )
    parser.add_argument(
        "pdb_directory",
        help="Path to the directory containing all .pdb files."
    )
    parser.add_argument(
        "output_fasta",
        help="Path to the output FASTA file to write (e.g. aggregated_unique.fasta)."
    )
    parser.add_argument(
        "--chain",
        dest="chain_id",
        default=None,
        help="(Optional) Chain ID to extract from each PDB. If omitted, the first chain will be used."
    )
    args = parser.parse_args()

    main(args.pdb_directory, args.output_fasta, chain_id=args.chain_id)

