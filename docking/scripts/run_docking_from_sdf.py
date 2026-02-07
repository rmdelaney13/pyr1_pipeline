#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
from configparser import ConfigParser

import docking_pipeline_utils as dpu


def _resolve_mode(cli_mode, config):
    if cli_mode != "auto":
        return cli_mode
    mode = dpu.cfg_get(config, "pipeline", "DockingMode", None)
    if mode is None:
        mode = dpu.cfg_get(config, "DEFAULT", "DockingMode", "glycine")
    mode = str(mode).strip().lower()
    aliases = {
        "glycine_shaved": "glycine",
        "glycine-shaved": "glycine",
        "glycine": "glycine",
        "sequence": "sequence",
        "docked_to_sequence": "sequence",
    }
    return aliases.get(mode, mode)


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["auto", "glycine", "sequence"],
        default="auto",
        help="Docking mode to run",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=0,
        help="Array index passed through to docking script",
    )
    parser.add_argument(
        "--skip-create-table",
        action="store_true",
        help="Skip create_table step and assume CSV already exists",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run create_table only and exit",
    )
    args = parser.parse_args(argv)

    config = ConfigParser()
    with open(args.config_file, "r", encoding="utf-8-sig") as handle:
        config.read_file(handle)

    csv_file_name = dpu.cfg_get(config, "create_table", "CSVFileName", dpu.cfg_get(config, "DEFAULT", "CSVFileName"))
    if not args.skip_create_table:
        dpu.ensure_table_ready(args.config_file, csv_file_name)

    if args.prepare_only:
        print(f"Preparation complete. CSV available: {csv_file_name}")
        return

    mode = _resolve_mode(args.mode, config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_map = {
        "glycine": os.path.join(script_dir, "grade_conformers_glycine_shaved_docking_multiple_slurm.py"),
        "sequence": os.path.join(script_dir, "grade_conformers_docked_to_sequence_multiple_slurm1.py"),
    }
    if mode not in script_map:
        raise ValueError(
            f"Unsupported docking mode '{mode}'. Expected one of: {', '.join(script_map.keys())}"
        )

    cmd = [sys.executable, script_map[mode], args.config_file, str(args.array_index)]
    print(f"Running docking mode '{mode}': {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
