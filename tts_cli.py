"""Isolated Fish Speech (S2) two-stage generation CLI (random-voice path).

Stage 1: text -> semantic codes (text2semantic).
Stage 2: semantic codes -> waveform (DAC decoder).

Runs in its own venv. tilde's main server invokes this as a subprocess.
Usage: python tts_cli.py --text "..." --out out.wav [--device mps|cpu]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
CKPT = REPO / "checkpoints" / "s2-pro"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if not CKPT.exists():
        print(f"Fish checkpoints missing: {CKPT}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="fish_"))

    # Stage 1: text -> codes_0.npy
    r1 = subprocess.run(
        [
            sys.executable, "fish_speech/models/text2semantic/inference.py",
            "--text", args.text,
            "--checkpoint-path", str(CKPT),
            "--device", args.device,
            "--num-samples", "1",
            "--output-dir", str(work),
        ],
        cwd=str(REPO), capture_output=True,
    )
    codes = work / "codes_0.npy"
    if r1.returncode != 0 or not codes.exists():
        print(r1.stderr.decode(errors="replace")[-600:], file=sys.stderr)
        return 1

    # Stage 2: codes -> wav
    r2 = subprocess.run(
        [
            sys.executable, "fish_speech/models/dac/inference.py",
            "-i", str(codes),
            "-o", args.out,
            "--checkpoint-path", str(CKPT / "codec.pth"),
            "--device", args.device,
        ],
        cwd=str(REPO), capture_output=True,
    )
    if r2.returncode != 0 or not Path(args.out).exists():
        print(r2.stderr.decode(errors="replace")[-600:], file=sys.stderr)
        return 1

    print(f"OK {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
