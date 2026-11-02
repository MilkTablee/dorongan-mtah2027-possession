"""Label the provided game and write a submission, printing aggregates only.

    python -m possession.predict --name our_team
"""

import argparse
import json
import os
import sys

import numpy as np

from .data import ROOT, load_provided
from .model import GAP_FRAMES, features, predict
from .tune import PARAMS_PATH

sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import write_submission  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def report(labels):
    """Possession counts and durations: enough to debug by, and no frame-level data."""
    graded = labels[labels != "X"]
    edges = np.flatnonzero(graded[1:] != graded[:-1]) + 1
    runs = np.split(graded, edges)
    seconds = {letter: np.array([len(r) / 25 for r in runs if r[0] == letter] or [0.0])
               for letter in "HAD"}
    print("graded frames {}".format(len(graded)))
    for letter in "HAD":
        print("  {}: {:5.1%} of frames, {:4d} spells, mean {:6.1f} s, median {:6.1f} s, "
              "under 2 s {}".format(letter, float(np.mean(graded == letter)),
                                    int((seconds[letter] > 0).sum()),
                                    float(seconds[letter].mean()),
                                    float(np.median(seconds[letter])),
                                    int((seconds[letter] < 2).sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession_v2")
    args = parser.parse_args()

    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    match = load_provided()
    labels = predict(match, features(match, params.get("gap_frames", GAP_FRAMES)), params)
    report(labels)
    path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
