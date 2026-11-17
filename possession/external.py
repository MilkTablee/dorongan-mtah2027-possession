"""Seven open Bundesliga matches (IDSSE, Bassek et al. 2025, CC-BY) as training and test data.

These matches carry per-frame ball possession and ball status from corrected tracking, which is the
same kind of label as the challenge ground truth. That makes them better training data than labels
rebuilt from event data, and a test bench whose numbers track the real board: nearest-player
smoothing scores close to its published 0.2433 floor here once the tracking is degraded.

The XML files are downloaded by hand into IDSSE_DIR (see README) and parsed once into a cache.
"""

import os

import numpy as np

from .data import ROOT, FPS, Match, _degrade

IDSSE_DIR = os.environ.get("MTAH_IDSSE_DIR") or os.path.join(os.path.dirname(ROOT), "idsse")
IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def _files(match_id):
    names = os.listdir(IDSSE_DIR)
    meta = [n for n in names if match_id in n and "matchinformation" in n.lower() and n.endswith(".xml")]
    raw = [n for n in names if match_id in n and "position" in n.lower() and n.endswith(".xml")]
    if len(meta) != 1 or len(raw) != 1:
        raise FileNotFoundError("{}: expected one meta and one positions XML in {}".format(
            match_id, IDSSE_DIR))
    return os.path.join(IDSSE_DIR, meta[0]), os.path.join(IDSSE_DIR, raw[0])


def _parse(match_id):
    """Positions in metres, team of each slot, period index ranges and H/A/D labels."""
    from kloppy import sportec

    meta, raw = _files(match_id)
    dataset = sportec.load_tracking(meta_data=meta, raw_data=raw, coordinates="kloppy",
                                    only_alive=False)
    dims = dataset.metadata.pitch_dimensions
    length, width = float(dims.pitch_length or 105.0), float(dims.pitch_width or 68.0)
    home, away = dataset.metadata.teams
    roster = list(home.players) + list(away.players)
    slot = {p.player_id: i for i, p in enumerate(roster)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)

    frames = dataset.frames
    ball = np.full((len(frames), 2), np.nan, dtype=np.float32)
    players = np.full((len(frames), len(roster), 2), np.nan, dtype=np.float32)
    period = np.zeros(len(frames), dtype=np.int16)
    labels = np.full(len(frames), "D", dtype="<U1")
    owner, current = "D", None
    for t, frame in enumerate(frames):
        period[t] = frame.period.id
        if frame.period.id != current:  # possession never carries across a half
            current, owner = frame.period.id, "D"
        if frame.ball_coordinates is not None:
            ball[t] = (frame.ball_coordinates.x * length, frame.ball_coordinates.y * width)
        for player, data in frame.players_data.items():
            if data.coordinates is not None and player.player_id in slot:
                players[t, slot[player.player_id]] = (data.coordinates.x * length,
                                                      data.coordinates.y * width)
        if frame.ball_owning_team is not None:
            owner = "H" if frame.ball_owning_team is home else "A"
        alive = frame.ball_state is not None and frame.ball_state.value == "alive"
        labels[t] = owner if alive and owner != "D" else "D"

    # Our features assume 105 x 68; rescale so distances and the out-of-play test line up.
    scale = np.array([105.0 / length, 68.0 / width], dtype=np.float32)
    return ball * scale, players * scale, team, period, labels


def load_idsse(match_id, seed=0, long_gaps=True):
    """One IDSSE match, degraded to look like the provided automatic tracking."""
    cache = os.path.join(IDSSE_DIR, "cache", match_id + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        ball, players, team, period, labels = (z["ball"], z["players"], z["team"], z["period"],
                                               z["labels"])
    else:
        ball, players, team, period, labels = _parse(match_id)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, ball=ball, players=players, team=team, period=period,
                            labels=labels)

    edges = np.flatnonzero(np.diff(period)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(period)]
    runs = [(int(a), int(b) - 1) for a, b in zip(starts, stops)]
    live = labels != "D"
    restarts = list(np.flatnonzero(live[1:] & ~live[:-1]) + 1)

    seed_offset = IDSSE_MATCHES.index(match_id) + 10
    ball, players, tags = _degrade(ball.astype(float), players.astype(float), team, restarts,
                                   len(ball), seed + seed_offset, long_gaps=long_gaps)
    return Match("idsse_" + match_id, ball.astype(np.float32), players.astype(np.float32),
                 team, runs, labels, tags)


if __name__ == "__main__":
    # Parse every match into the cache, in parallel, and print a summary.
    import sys
    from multiprocessing import Pool

    with Pool(min(4, os.cpu_count() or 1)) as pool:
        matches = pool.map(load_idsse, sys.argv[1:] or IDSSE_MATCHES)
    for m in matches:
        graded = m.labels != "X"
        print("{}: {} frames, {} halves, ball seen {:.1%}, H {:.1%} A {:.1%} D {:.1%}".format(
            m.name, len(m.labels), len(m.runs), float(np.mean(~np.isnan(m.ball[graded, 0]))),
            *[float(np.mean(m.labels == x)) for x in "HAD"]))
