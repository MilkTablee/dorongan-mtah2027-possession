"""Tune the decoder on Metrica's open sample games, never on the provided game.

Random search, scored with the official spellF1. Each game is held out in turn: parameters are
picked on the other two and scored on it, so the printed figure is an honest estimate. The final
parameters are then picked on all three and written to possession/params.json.

    python -m possession.tune --trials 300
"""

import argparse
import json
import os
import sys

import numpy as np

from .data import HERE, ROOT, load_sample
from .model import features, predict

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from baseline import smooth  # noqa: E402

PARAMS_PATH = os.path.join(HERE, "params.json")

# The tag term stays off: simulated tags did not resemble the real ones closely enough to tune it.
FIXED = {"w_tag": 0.0}

# (low, high, log scale). Ranges are football judgement, not fitted to anything.
SPACE = {
    "radius": (1.5, 6.0, False), "margin": (1.0, 6.0, False), "fast": (6.0, 20.0, False),
    "flight": (0.0, 1.0, False), "w_team": (0.05, 2.0, True), "still": (0.4, 1.2, False),
    "w_still": (0.02, 2.0, True), "bias": (0.0, 1.0, False), "w_out": (0.05, 3.0, True),
    "w_kickoff": (0.02, 2.0, True), "w_fast": (0.01, 1.0, True),
    "swap": (5.0, 400.0, True), "stop": (5.0, 400.0, True), "start": (5.0, 400.0, True),
}


def draw(rng):
    out = dict(FIXED)
    for name, (low, high, log) in SPACE.items():
        value = np.exp(rng.uniform(np.log(low), np.log(high))) if log else rng.uniform(low, high)
        out[name] = float(value)
    return out


def nudge(rng, params, scale=0.2):
    """A nearby point, clipped to the search space."""
    out = dict(params)
    for name, (low, high, _) in SPACE.items():
        out[name] = float(np.clip(params[name] * np.exp(rng.normal(0, scale)), low, high))
    return out


def spell_score(labels, predicted):
    gt = pm.label_codes(labels)
    return pm.evaluate(gt, pm.label_codes(predicted))["spell_f1_mean"]


def baseline_score(match):
    """The published floor's algorithm on the same degraded data, for comparison."""
    distance = np.sqrt(((match.players - match.ball[:, None]) ** 2).sum(axis=2))
    distance = np.where(np.isnan(distance), np.inf, distance)
    codes = match.team[distance.argmin(axis=1)].astype(np.int8)
    codes[np.isnan(match.ball[:, 0])] = -1
    return spell_score(match.labels, smooth(codes, 50))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--refine", type=int, default=150,
                        help="extra trials around the best of the random search")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matches = [load_sample(number, args.seed) for number in (1, 2, 3)]
    for match in matches:
        labels = match.labels
        graded = int((labels != "X").sum())
        spells = pm.possession_spells(pm.label_codes(labels), pm.valid_runs(pm.label_codes(labels)))
        print("{}: {} graded frames, {} runs, ball seen {:.1%}, D {:.1%}, {} possessions, "
              "mean {:.1f} s, baseline_smooth50 {:.4f}".format(
                  match.name, graded, len(match.runs),
                  float(np.mean(~np.isnan(match.ball[labels != "X", 0]))),
                  float(np.mean(labels[labels != "X"] == "D")), len(spells),
                  graded * float(np.mean(labels[labels != "X"] != "D")) / max(len(spells), 1) / 25,
                  baseline_score(match)), flush=True)

    cached = [features(match) for match in matches]
    rng = np.random.default_rng(args.seed)
    trials, rows = [], []

    def run(params):
        trials.append(params)
        rows.append([spell_score(match.labels, predict(match, f, params))
                     for match, f in zip(matches, cached)])
        if len(trials) % 25 == 0:
            print("trial {}: best mean so far {:.4f}".format(
                len(trials), np.mean(rows, axis=1).max()), flush=True)

    for _ in range(args.trials):
        run(draw(rng))
    for _ in range(args.refine):
        run(nudge(rng, trials[int(np.mean(rows, axis=1).argmax())]))
    scores = np.array(rows)

    held_out = []
    for j, match in enumerate(matches):
        others = [k for k in range(len(matches)) if k != j]
        pick = int(scores[:, others].mean(axis=1).argmax())
        held_out.append(scores[pick, j])
        print("held out {}: spellF1 {:.4f}".format(match.name, scores[pick, j]))
    print("mean held-out spellF1 {:.4f}".format(float(np.mean(held_out))))

    best = int(scores.mean(axis=1).argmax())
    print("final parameters, mean on all three {:.4f}".format(scores[best].mean()))
    print(json.dumps(trials[best], indent=2))
    with open(PARAMS_PATH, "w") as handle:
        json.dump(trials[best], handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
