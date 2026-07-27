#!/usr/bin/env python3
"""Retrain templates + Stage-2 heads after preprocessing pipeline changes.

Runs the full bio sync path:
  1. phase2a_enroll  — rebuild face/voice templates from enroll samples
  2. train_face_pad  — PAD head (hand-crafted features)
  3. train_face_calibrator
  4. train_voice_calibrator

Usage:
  python scripts/retrain_bio_pipeline.py --store driveauth_store_phase2a \\
      --data data/driver1 --driver-id driver1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["DRIVEAUTH_STAGE2_RAW"] = "1"
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "driveauth_store_phase2a")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "driver1")
    ap.add_argument("--driver-id", default="driver1")
    ap.add_argument("--skip-enroll", action="store_true")
    ap.add_argument(
        "--exclude-fallback-crops",
        action="store_true",
        help="Pass through to train_face_pad.py",
    )
    args = ap.parse_args()

    py = sys.executable
    store = str(args.store)
    data = str(args.data)
    did = args.driver_id

    if not args.skip_enroll:
        _run(
            [
                py,
                str(ROOT / "scripts" / "phase2a_enroll.py"),
                "--store",
                store,
                "--data",
                data,
                "--driver-id",
                did,
            ]
        )

    pad_cmd = [
        py,
        str(ROOT / "scripts" / "train_face_pad.py"),
        "--store",
        store,
        "--data",
        data,
        "--driver-id",
        did,
    ]
    if args.exclude_fallback_crops:
        pad_cmd.append("--exclude-fallback-crops")
    _run(pad_cmd)

    _run(
        [
            py,
            str(ROOT / "scripts" / "train_face_calibrator.py"),
            "--store",
            store,
            "--data",
            data,
            "--driver-id",
            did,
        ]
    )
    _run(
        [
            py,
            str(ROOT / "scripts" / "train_voice_calibrator.py"),
            "--store",
            store,
            "--data",
            data,
            "--driver-id",
            did,
        ]
    )

    _run([py, str(ROOT / "scripts" / "bootstrap.py"), "--check-only", "--store", store])
    print("\nBio retrain pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
