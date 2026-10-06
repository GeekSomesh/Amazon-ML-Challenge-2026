"""Assemble <team_name>_submission.zip in the structure required by the challenge:

<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt, run_pipeline.sh}
└── Documentation_template.md

  python package_submission.py <team_name> <documentation.md> [destination_dir]
"""
import os
import sys
import zipfile

from config import OUTPUT

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(team, doc_path, dest_dir):
    zpath = os.path.join(dest_dir, f"{team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(OUTPUT, f), f"output/{f}")
        for f in ("README.md", "requirements.txt", "run_pipeline.sh"):
            z.write(os.path.join(CODE, f), f"code/business_entity_resolution/{f}")
        src = os.path.join(CODE, "src")
        for d, dirs, files in os.walk(src):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                if f.endswith(".py"):
                    full = os.path.join(d, f)
                    rel = os.path.relpath(full, CODE).replace(os.sep, "/")
                    z.write(full, f"code/business_entity_resolution/{rel}")
        z.write(doc_path, "Documentation_template.md")
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(OUTPUT))
