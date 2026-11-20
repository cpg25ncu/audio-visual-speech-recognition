"""Command-line interface: ``avsr prepare | run | report | predict``."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GLOG_minloglevel", "2")       # silence MediaPipe's C++ logging


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="avsr", description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="manifest, aligned audio cache and mouth tracks")
    p.add_argument("--methods", nargs="+", default=["landmarks", "haar"])
    p.add_argument("--force", action="store_true")

    r = sub.add_parser("run", help="run experiments")
    r.add_argument("experiments", nargs="+",
                   choices=["all", "audio", "audio_frontend", "visual", "dct_sweep", "confound", "fusion", "fusion_late",
                            "framework_parity", "deployment"])
    r.add_argument("--seeds", nargs="+", type=int, default=None)
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--backend", choices=["torch", "keras"], default=None,
                   help="deep-learning framework (default: training.backend in the config, i.e. torch)")

    sub.add_parser("report", help="figures and RESULTS.md from the result tables")

    d = sub.add_parser("predict", help="recognise the name spoken in a video clip")
    d.add_argument("video")
    d.add_argument("--noise", choices=["white", "pink", "ambient"], default=None)
    d.add_argument("--snr", type=float, default=None)
    d.add_argument("--top", type=int, default=3)

    args = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)

    if args.cmd == "prepare":
        from .prepare import prepare_all
        prepare_all(cfg, methods=args.methods, force=args.force)
    elif args.cmd == "run":
        from .experiments import run
        run(cfg, args.experiments, seeds=args.seeds, epochs=args.epochs, backend=args.backend)
    elif args.cmd == "report":
        from .report import build_report
        build_report(cfg)
    elif args.cmd == "predict":
        from .inference import Predictor
        out = Predictor(cfg).predict(args.video, noise=args.noise, snr=args.snr, top=args.top)
        for stream, ranked in out.items():
            print(f"{stream:>14}: " + ", ".join(f"{name} ({p:.0%})" for name, p in ranked))


if __name__ == "__main__":
    main()
