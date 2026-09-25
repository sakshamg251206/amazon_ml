"""Pipeline entry point. Phase 1: --stage rule."""
import argparse
import csv
import time
from datetime import datetime

from src import config
from src.baseline_rule import rule_matches
from src.data import read_ground_truth, read_source
from src.decide import write_id_lists
from src.metrics import macro_f05

LOG = config.ROOT / "experiments" / "log.csv"


def log_result(version: str, metrics: dict[str, float]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "version", "seed", "metric", "value"])
        for k, v in metrics.items():
            w.writerow([datetime.now().isoformat(timespec="seconds"), version, config.SEED, k, f"{v:.5f}"])


def run_rule(split: str) -> None:
    t0 = time.time()
    s1 = read_source(split, 1)
    matches: dict[str, list[str]] = {}
    for src in (2, 3):  # one source at a time keeps peak memory low on an 8 GB machine
        for k, v in rule_matches(s1, read_source(split, src)).items():
            matches.setdefault(k, []).extend(v)
    print(f"rule matched {len(matches):,} S1 in {time.time() - t0:.0f}s")

    if split == "train":
        truth = read_ground_truth()
        pred = {k: set(v) for k, v in matches.items()}
        metrics = {"f05_all": macro_f05(pred, truth)}
        for country, ids in s1.group_by("country").agg("entity_id").iter_rows():
            metrics[f"f05_{country}"] = macro_f05(pred, {i: truth[i] for i in ids})
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        log_result("v0_rule", metrics)
    else:
        ids = s1["entity_id"].to_list()
        write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids", ids, matches)
        # For the rule baseline the "model" is the rule, so candidates == matches.
        write_id_lists(config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", ids, matches)
        print(f"wrote {config.OUTPUT_DIR}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["rule"])
    p.add_argument("--split", required=True, choices=["train", "test"])
    a = p.parse_args(argv)
    if a.stage == "rule":
        run_rule(a.split)


if __name__ == "__main__":
    main()
