"""Command line: python -m resolver <command>

  synth     generate the synthetic dataset (challenge layout)
  train     fit the two-stage matcher on a labelled split, with an out-of-fold report
  evaluate  score a labelled split with a fitted model
  resolve   write matching_results.tsv + candidate_pairs.tsv for any split (validated)
  demo      everything the web app needs: synth (if missing) -> train -> evaluate -> index
  serve     run the web app / API
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from resolver import config

log = logging.getLogger("resolver")


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def cmd_synth(a) -> None:
    from resolver.synth.generator import SynthConfig, generate
    stats = generate(a.out, SynthConfig(seed=a.seed).scaled(a.scale))
    print(json.dumps(stats, indent=1))


def cmd_train(a) -> None:
    from resolver.io import Dataset
    from resolver.train import fit
    matcher, report = fit(Dataset.load(a.data), k_folds=a.folds, workers=a.workers)
    matcher.save(a.model)
    _dump(report, Path(a.report))
    f = report["final"]
    print(f"out-of-fold macro F0.5 {f['f05']:.4f}  precision {f['precision']:.4f}  recall {f['recall']:.4f}"
          f"  -> {a.model}")


def cmd_evaluate(a) -> None:
    from resolver.engine import Matcher
    from resolver.io import Dataset
    from resolver.train import evaluate
    _, report = evaluate(Dataset.load(a.data), Matcher.load(a.model), workers=a.workers)
    _dump(report, Path(a.report))
    if "final" in report:
        f = report["final"]
        print(f"macro F0.5 {f['f05']:.4f}  precision {f['precision']:.4f}  recall {f['recall']:.4f}")
        for c in report["slices"]["by_country"]:
            print(f"  {c['country']:8s} F0.5 {c['f05']:.4f}")


def cmd_resolve(a) -> None:
    from resolver.engine import Matcher, resolve
    from resolver.io import Dataset, submission_tables, validate_submission, write_submission
    ds = Dataset.load(a.data)
    res = resolve(ds, Matcher.load(a.model), workers=a.workers)
    paths = write_submission(a.out, ds.s1["entity_id"], res.matches, res.candidates)
    t = submission_tables(ds.s1["entity_id"], res.matches, res.candidates)
    issues = validate_submission(t["matching_results.tsv"], t["candidate_pairs.tsv"], ds)
    print("\n".join(str(p) for p in paths))
    print("validation:", "PASS" if not issues else "; ".join(issues))
    sys.exit(1 if any(not i.startswith("warning") for i in issues) else 0)


def cmd_demo(a) -> None:
    from resolver.engine import Matcher
    from resolver.index import ResolverIndex
    from resolver.io import Dataset
    from resolver.synth.generator import SynthConfig, generate
    from resolver.train import evaluate, fit
    import joblib
    t0 = time.time()
    data = Path(a.data)
    if a.regenerate or not (data / "train" / "train_source1.tsv").exists():
        log.info("generating synthetic data -> %s", data)
        generate(data, SynthConfig(seed=a.seed).scaled(a.scale))
    if a.retrain or not config.MODEL_PATH.exists():
        matcher, report = fit(Dataset.load(data / "train"), workers=a.workers)
        matcher.save(config.MODEL_PATH)
        _dump(report, config.TRAIN_REPORT)
    matcher = Matcher.load(config.MODEL_PATH)
    test = Dataset.load(data / "test")
    _, report = evaluate(test, matcher, workers=a.workers)
    _dump(report, config.TEST_REPORT)
    index = ResolverIndex.build(test, matcher, workers=a.workers)
    joblib.dump(index, config.INDEX_PATH, compress=3)
    f = report.get("final", {})
    print(f"demo ready in {time.time() - t0:.0f}s: test macro F0.5 {f.get('f05', float('nan')):.4f} -> {config.ARTIFACTS}")


def cmd_serve(a) -> None:
    import uvicorn
    uvicorn.run("resolver.api.app:app", host=a.host, port=a.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="python -m resolver", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate synthetic data")
    s.add_argument("--out", default=str(config.DEMO_DATA))
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=42)
    for name, helptext in (("train", "fit the matcher"), ("evaluate", "score a labelled split"),
                           ("resolve", "write submission files")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--data", required=True, help="directory with *_source{1,2,3}.tsv")
        c.add_argument("--model", default=str(config.MODEL_PATH))
        c.add_argument("--workers", type=int, default=1, help="processes for text normalisation")
        if name == "train":
            c.add_argument("--folds", type=int, default=2)
            c.add_argument("--report", default=str(config.TRAIN_REPORT))
        if name == "evaluate":
            c.add_argument("--report", default=str(config.TEST_REPORT))
        if name == "resolve":
            c.add_argument("--out", default="output")
    d = sub.add_parser("demo", help="build everything the web app needs")
    d.add_argument("--data", default=str(config.DEMO_DATA))
    d.add_argument("--scale", type=float, default=1.0)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--regenerate", action="store_true")
    d.add_argument("--retrain", action="store_true")
    v = sub.add_parser("serve", help="run the web app")
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    {"synth": cmd_synth, "train": cmd_train, "evaluate": cmd_evaluate, "resolve": cmd_resolve,
     "demo": cmd_demo, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
