from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = ROOT / "submission"

KIT_TOOLS = Path(__file__).resolve().parent / "starting_kit_tools"


SKIP_PARTS = {"__pycache__", ".DS_Store", ".pytest_cache", ".ipynb_checkpoints"}


def build_zip(submission_dir: Path, out: Path) -> None:
    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission dir not found: {submission_dir}")
    if not (submission_dir / "model.py").exists():
        raise FileNotFoundError(f"model.py not found in {submission_dir}")


    if out.exists():
        out.unlink()
    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(submission_dir.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            arcname = path.relative_to(submission_dir).as_posix()
            zf.write(path, arcname=arcname)
            n_files += 1
    print(f"[package] wrote {out}  ({n_files} files, "
          f"{out.stat().st_size / 1024:.1f} KB)")


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n[package] === {label} ===")
    print(f"[package] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"[package] {label} failed (exit {proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output ZIP path. Defaults to <repo>/<submission-dir-name>.zip "
                             "(e.g. submission/ -> submission.zip, "
                             "submission_full/ -> submission_full.zip).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip run_smoke_test.py (e.g., on a node without sample_data).")
    parser.add_argument("--skip-zip-check", action="store_true",
                        help="Skip check_submission_zip.py.")
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / f"{args.submission_dir.resolve().name}.zip"

    smoke = KIT_TOOLS / "run_smoke_test.py"
    zipcheck = KIT_TOOLS / "check_submission_zip.py"


    if not args.skip_smoke:
        if not smoke.exists():
            raise FileNotFoundError(f"Missing tool: {smoke}")
        run_step("smoke test (run_smoke_test.py)",
                 [sys.executable, str(smoke), str(args.submission_dir)])


    print()
    build_zip(args.submission_dir, args.out)


    if not args.skip_zip_check:
        if not zipcheck.exists():
            raise FileNotFoundError(f"Missing tool: {zipcheck}")
        run_step("zip layout check (check_submission_zip.py)",
                 [sys.executable, str(zipcheck), str(args.out)])

    print(f"\n[package] ALL CHECKS PASSED -- ready to upload {args.out}")


if __name__ == "__main__":
    main()
