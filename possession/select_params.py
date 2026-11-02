"""Choose between tuned parameter sets on a simulation seed none of them was tuned on.

    python -m possession.select_params --candidates seed0.json seed1.json

Both candidates come from possession.tune. Whichever scores better on the fresh seed is written to
possession/params.json, which is the only file possession.predict reads.
"""

import argparse
import json

import numpy as np

from .data import load_sample
from .model import features, predict
from .tune import PARAMS_PATH, spell_score

SELECTION_SEED = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    matches = [load_sample(number, SELECTION_SEED) for number in (1, 2, 3)]
    cached = [features(match, 0) for match in matches]

    best, best_score = None, -1.0
    for path in args.candidates:
        with open(path) as handle:
            params = json.load(handle)
        score = float(np.mean([spell_score(match.labels, predict(match, f, params))
                               for match, f in zip(matches, cached)]))
        print("{}: spellF1 {:.4f} on seed {}".format(path, score, SELECTION_SEED))
        if score > best_score:
            best, best_score = params, score

    best["gap_frames"] = 0  # the chosen model runs without ball-gap bridging
    with open(PARAMS_PATH, "w") as handle:
        json.dump(best, handle, indent=2)
        handle.write("\n")
    print("wrote {}".format(PARAMS_PATH))


if __name__ == "__main__":
    main()
