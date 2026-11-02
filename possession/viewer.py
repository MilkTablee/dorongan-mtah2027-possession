"""Watch a submission over the tactical video, on your own machine.

    pip install opencv-python
    python -m possession.viewer evaluation/results/possession_v1.csv --start 20000

Keys: space pauses, n and p jump to the next or previous change of label, q quits.

For debugging only. RULES.md allows watching the video to understand a system, but what you see
may not become labels, corrections, or rules tied to specific frames. Fixes must be general logic.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .data import ROOT

sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

COLOURS = {"H": (160, 60, 0), "A": (40, 40, 220), "D": (120, 120, 120), "X": (0, 0, 0)}


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--start", type=int, default=11185, help="first video frame to show")
    args = parser.parse_args()

    labels = pd.read_csv(args.submission)["label"].to_numpy()
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    video = cv2.VideoCapture(asset_paths()["video"])
    video.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame, paused = args.start, False
    image = None

    while True:
        if not paused or image is None:
            ok, image = video.read()
            if not ok:
                break
            frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        shown = image.copy()
        label = labels[frame]
        cv2.rectangle(shown, (0, 0), (360, 48), COLOURS[label], -1)
        cv2.putText(shown, "{}  frame {}".format(label, frame), (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(os.path.basename(args.submission), shown)

        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("n"), ord("p")):
            ahead = changes[changes > frame + 12]
            behind = changes[changes < frame - 12]
            target = ahead[0] if key == ord("n") and len(ahead) else (
                behind[-1] if key == ord("p") and len(behind) else frame)
            video.set(cv2.CAP_PROP_POS_FRAMES, max(int(target) - 50, 0))
            image = None

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
