"""A learned frame classifier feeding the same Viterbi decoder.

The hand-set evidence in model.py uses a handful of signals at one time scale. Here a gradient
boosted classifier sees many signals at several scales and gives each frame a probability for H, A
and D. Those probabilities replace the hand-set evidence, and decoding is unchanged.

Trained only on Metrica's open sample games, degraded to look like the provided tracking. Home and
away are mirrored during training, so the model cannot learn that one side has the ball more.

    python -m possession.learned --evaluate     held-out scores on the sample games
    python -m possession.learned --name Dorongan_v3
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import HERE, ROOT, load_provided, load_sample
from .model import FPS, LETTERS, PITCH, absorb_short_dead, features, viterbi
from .tune import spell_score

sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import write_submission  # noqa: E402

PARAMS_PATH = os.path.join(HERE, "learned_params.json")
WINDOWS = (1, 3, 8)  # seconds
SUBSAMPLE = 5        # train on every 5th frame; neighbours are nearly identical
SEED = 0
COSTS = [{"swap": s, "stop": t, "start": t} for s in (10, 20, 40, 80, 160) for t in (6, 12, 24, 48, 96)]


def _mean(x, seconds):
    """Centred moving average ignoring NaN, and the share of the window that was not NaN."""
    width = int(seconds * FPS) | 1
    seen = ~np.isnan(x)
    total = np.convolve(np.where(seen, x, 0.0), np.ones(width), mode="same")
    count = np.convolve(seen.astype(float), np.ones(width), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count, count / width


def table(match, f):
    """(T, n) features, and the same with home and away swapped."""
    ball = match.ball
    length, width = PITCH
    with np.errstate(invalid="ignore"):
        to_line = np.fmin(np.fmin(ball[:, 0], length - ball[:, 0]),
                          np.fmin(ball[:, 1], width - ball[:, 1]))
        to_goal_line = np.fmin(ball[:, 0], length - ball[:, 0])
    seen = (~np.isnan(ball[:, 0])).astype(float)
    margin = f["away"] - f["home"]

    symmetric = {"ball_speed": f["ball_speed"], "motion": f["motion"].astype(float),
                 "outside": f["outside"].astype(float), "kickoff": f["kickoff"].astype(float),
                 "seen": seen, "to_line": to_line,
                 "to_goal_line": to_goal_line, "closest": np.fmin(f["home"], f["away"])}
    signed = {"margin": margin, "spread": f["spread"].astype(float)}

    def build(sign):
        columns = []
        for values in symmetric.values():
            columns.append(values)
            for seconds in WINDOWS:
                mean, share = _mean(values, seconds)
                columns += [mean, share]
        for values in signed.values():
            columns.append(sign * values)
            for seconds in WINDOWS:
                mean, share = _mean(sign * values, seconds)
                columns += [mean, share]
        return np.stack(columns, axis=1).astype(np.float32)

    return build(1.0), build(-1.0)


def training_rows(matches, cached):
    xs, ys = [], []
    for match, f in zip(matches, cached):
        straight, mirrored = table(match, f)
        graded = np.flatnonzero(match.labels != "X")[::SUBSAMPLE]
        label = match.labels[graded]
        flipped = np.where(label == "H", "A", np.where(label == "A", "H", label))
        xs += [straight[graded], mirrored[graded]]
        ys += [label, flipped]
    return np.concatenate(xs), np.concatenate(ys)


def fit(matches, cached):
    x, y = training_rows(matches, cached)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    return model.fit(x, y)


def log_probs(model, match, f):
    """(T, 3) log-probabilities in H, A, D order, averaged over both team orientations."""
    straight, mirrored = table(match, f)
    order = [list(model.classes_).index(letter) for letter in LETTERS]
    p = model.predict_proba(straight)[:, order]
    q = model.predict_proba(mirrored)[:, order][:, [1, 0, 2]]
    return np.log((p + q) / 2 + 1e-4)


def decode(match, score, costs):
    out = np.full(len(match.ball), "X", dtype="<U1")
    for first, last in match.runs:
        out[first:last + 1] = LETTERS[absorb_short_dead(viterbi(score[first:last + 1], costs))]
    return out


def load(source, seed):
    """Training matches: Metrica's three samples, or the seven IDSSE matches."""
    if source == "idsse":
        from .external import IDSSE_MATCHES, load_idsse
        return [load_idsse(match_id, seed) for match_id in IDSSE_MATCHES]
    return [load_sample(number, seed) for number in (1, 2, 3)]


def extra_copies(source, seed, augment):
    """Extra degraded copies of each training match, or None.

    seeds: a second copy degraded with a different seed.
    mixed: a second copy with short, frequent ball gaps instead of the measured long ones, in case
    the measured pattern does not describe every way the real ball goes missing.
    """
    if source != "idsse" or augment == "none":
        return None
    from .external import IDSSE_MATCHES, load_idsse
    copies = [load_idsse(match_id, seed + 100, long_gaps=(augment == "seeds"))
              for match_id in IDSSE_MATCHES]
    return [(match, features(match, 0)) for match in copies]


def out_of_fold(matches, cached, extra=None):
    """Each match scored by a model trained on all the others, and their extra copies."""
    folds = []
    for k in range(len(matches)):
        rest = [m for m in range(len(matches)) if m != k]
        train = [matches[m] for m in rest] + [extra[m][0] for m in rest] if extra else             [matches[m] for m in rest]
        train_f = [cached[m] for m in rest] + [extra[m][1] for m in rest] if extra else             [cached[m] for m in rest]
        model = fit(train, train_f)
        folds.append((matches[k], log_probs(model, matches[k], cached[k])))
        print("  fold {} done".format(matches[k].name), flush=True)
    return folds


def pick_costs(folds):
    return max(COSTS, key=lambda c: np.mean([spell_score(m.labels, decode(m, s, c))
                                             for m, s in folds]))


def evaluate(source, seed, augment="none"):
    """Leave one match out, against the floor and the hand-set v2 model on the same matches.

    Decoder costs are one choice of three numbers made on the same out-of-fold predictions that
    are reported, so the learned figure is slightly optimistic.
    """
    from .model import predict as predict_hand
    from .tune import PARAMS_PATH as HAND_PARAMS, baseline_score

    matches = load(source, seed)
    cached = [features(match, 0) for match in matches]
    folds = out_of_fold(matches, cached, extra_copies(source, seed, augment))
    costs = pick_costs(folds)
    with open(HAND_PARAMS) as handle:
        hand = json.load(handle)

    rows = []
    for (match, score), f in zip(folds, cached):
        row = (baseline_score(match), spell_score(match.labels, predict_hand(match, f, hand)),
               spell_score(match.labels, decode(match, score, costs)))
        rows.append(row)
        print("{}: floor {:.4f}  v2 {:.4f}  learned {:.4f}".format(match.name, *row), flush=True)
    print("MEAN {} seed {} augment {}: floor {:.4f}  v2 {:.4f}  learned {:.4f}  costs {}".format(
        source, seed, augment, *np.mean(rows, axis=0), costs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--source", choices=("metrica", "idsse"), default="idsse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="possession_v3")
    parser.add_argument("--augment", choices=("none", "seeds", "mixed"), default="none",
                        help="train on a second degraded copy of every match as well")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.source, args.seed, args.augment)
        return

    matches = load(args.source, args.seed)
    cached = [features(match, 0) for match in matches]
    extra = extra_copies(args.source, args.seed, args.augment)
    costs = pick_costs(out_of_fold(matches, cached, extra))
    with open(PARAMS_PATH, "w") as handle:
        json.dump(costs, handle, indent=2)
        handle.write("\n")
    print("decoder costs {}".format(costs))

    model = fit(matches + [m for m, _ in extra or []], cached + [f for _, f in extra or []])
    match = load_provided()
    labels = decode(match, log_probs(model, match, features(match, 0)), costs)
    from .predict import report
    report(labels)
    path = write_submission(os.path.join(ROOT, "evaluation", "results", args.name + ".csv"), labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
