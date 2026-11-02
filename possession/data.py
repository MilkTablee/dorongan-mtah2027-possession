"""Load matches into one shape: the provided game, and Metrica's open sample games.

A Match holds ball positions and player positions in metres, the team of each track slot, and the
frame runs (halves) to decode. Sample games also carry labels derived from their event data. Those
labels are what the decoder is tuned on, because no label on the provided game may be used.

Sample data: https://github.com/metrica-sports/sample-data
"""

import json
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

SAMPLE_DIR = os.environ.get("MTAH_SAMPLE_DIR") or os.path.join(
    os.path.dirname(ROOT), "sample-data", "data")
CACHE_DIR = os.path.join(ROOT, "data", "cache")  # gitignored, like everything else under data/

PITCH = (105.0, 68.0)
FPS = 25
H, A, D, X = "H", "A", "D", "X"

# Hand-overs shorter than this are touches, not "controlled" possession (see CHALLENGE.md).
MIN_POSSESSION_FRAMES = int(1.5 * FPS)

# Chance per 10 s block of a home track and an away track trading identities in simulation.
ID_SWITCH_RATE = float(os.environ.get("MTAH_ID_SWITCH", "0.9"))

Match = namedtuple("Match", "name ball players team runs labels tags")
# ball (T, 2) m, NaN where unseen. players (T, N, 2) m. team (N) 0 home, 1 away.
# runs [(first, last)] inclusive. labels (T) of H/A/D/X, or None. tags (T) in [-1, 1], or None.


def _cache_path(name):
    return os.path.join(CACHE_DIR, name + ".npz")


def load_provided(game=GAME):
    """The match to label. Parsed once, then cached under data/, which git ignores."""
    from baseline import team_of_each_track
    from load_smart_tagging import load as load_tags

    path = _cache_path(game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    n_frames = source_frames("tactical", game)

    if os.path.exists(path):
        cached = np.load(path)
        ball, players = cached["ball"], cached["players"]
    else:
        ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
        players = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
        with open(raw_path) as handle:
            for line in handle:
                parts = line.split(":")
                if len(parts) != 3:
                    continue
                frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
                if parts[2][0] != "N":
                    ball[frame] = [float(v) for v in parts[2].split(",")[:2]]
                for slot, group in enumerate(parts[1].split(";")):
                    if group[0] != "N":
                        players[frame, slot] = [float(v) for v in group.split(",")[:2]]
        ball *= (length, width)
        players *= (length, width)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, ball=ball, players=players)

    runs = [(first, last) for _, first, last in half_video_blocks(game)]
    return Match(game, ball, players, team, runs, None, restart_signal(load_tags(game), n_frames))


def restart_signal(tags, n_frames):
    """+1 around where a tagged set piece restarts play, -1 once play has clearly resumed.

    Set-piece tags run about 35 s and begin a few seconds before the restart: mean player movement
    bottoms out at the start of the tag and has recovered about 8 s after it. So only the start
    of a tag carries timing, and the rest of the interval says nothing.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    for tag in tags:
        if tag["code"].strip().lower() != "set pieces":
            continue
        start = tag["start_frame"]
        out[max(start - 5 * FPS, 0):max(start + 2 * FPS, 0)] = 1.0
        out[max(start + 8 * FPS, 0):max(start + 16 * FPS, 0)] = -1.0
    return out


def _load_csv_game(number):
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))

    def side(name):
        frame = pd.read_csv(stem + "RawTrackingData_{}_Team.csv".format(name), skiprows=2)
        columns = list(frame.columns)
        slots = [i for i, c in enumerate(columns) if c.startswith("Player")]
        xy = np.stack([frame.iloc[:, [i, i + 1]].to_numpy(float) for i in slots], axis=1)
        where = columns.index("Ball")
        return xy, frame.iloc[:, [where, where + 1]].to_numpy(float)

    home, ball = side("Home")
    away, _ = side("Away")
    players = np.concatenate([home, away], axis=1)
    team = np.r_[np.zeros(home.shape[1], int), np.ones(away.shape[1], int)]

    raw = pd.read_csv(stem + "RawEventsData.csv")
    events = pd.DataFrame({
        "team": raw["Team"], "type": raw["Type"], "subtype": raw["Subtype"].fillna(""),
        "period": raw["Period"], "start": raw["Start Frame"] - 1, "end": raw["End Frame"] - 1})
    return ball, players, team, events


def _load_epts_game(number):
    from kloppy import metrica

    stem = os.path.join(SAMPLE_DIR, "Sample_Game_{0}/Sample_Game_{0}_".format(number))
    dataset = metrica.load_tracking_epts(meta_data=stem + "metadata.xml",
                                         raw_data=stem + "tracking.txt")
    frame = dataset.to_df()
    home, away = dataset.metadata.teams
    ids = [(p.player_id, 0) for p in home.players] + [(p.player_id, 1) for p in away.players]
    ids = [(pid, side) for pid, side in ids if "{}_x".format(pid) in frame.columns]
    players = np.stack([frame[["{}_x".format(p), "{}_y".format(p)]].to_numpy(float)
                        for p, _ in ids], axis=1)
    team = np.array([side for _, side in ids])
    ball = frame[["ball_x", "ball_y"]].to_numpy(float)

    # Index events by position in the tracking file, whatever its frame ids start from.
    offset = int(frame["frame_id"].iloc[0])
    with open(stem + "events.json", encoding="utf8") as handle:
        rows = json.load(handle)["data"]

    def subtype(entry):
        value = entry["subtypes"]
        if isinstance(value, dict):
            value = [value]
        return "-".join(item["name"] for item in value or [])

    events = pd.DataFrame({
        "team": ["Home" if e["team"]["name"] == "Team A" else "Away" for e in rows],
        "type": [e["type"]["name"] for e in rows],
        "subtype": [subtype(e) for e in rows],
        "period": [e["period"] for e in rows],
        "start": [e["start"]["frame"] - offset for e in rows],
        "end": [e["end"]["frame"] - offset for e in rows]})
    return ball, players, team, events


def labels_from_events(events, n_frames):
    """Frame labels from human event data, and the set-piece starts that end each stoppage.

    Each period spans its first to last event, and everything else stays X. The ball dies on a
    ball out, a shot that leaves the pitch or scores, a foul, or an offside; it lives again on the
    next restart. The owner is whoever last made a controlled action, and very short hand-overs
    are merged back, because a touch is not a possession. This approximates the challenge's
    definition, which is mechanical on corrected tracking we do not have for these games.
    """
    labels = np.full(n_frames, X, dtype="<U1")
    restarts = []

    for period, group in events.groupby("period"):
        group = group.sort_values("start", kind="stable")
        first, last = int(group["start"].min()), int(group["end"].max())
        owner = np.full(n_frames, -1, dtype=np.int8)
        dead = np.zeros(n_frames, dtype=bool)
        dead_since, current = None, -1

        for event in group.itertuples():
            side = 0 if event.team == "Home" else 1
            subtype = str(event.subtype).upper()
            restart = event.type == "SET PIECE" or "GOAL KICK" in subtype
            if restart:
                restarts.append(int(event.start))
                if dead_since is not None:
                    dead[dead_since:event.start] = True
                    dead_since = None
                current = side
            if event.type in ("PASS", "SHOT", "RECOVERY", "CARRY", "SET PIECE"):
                current = side
            owner[event.start:] = current

            stops = (event.type in ("BALL OUT", "FAULT RECEIVED")
                     or (event.type == "SHOT" and ("-OUT" in subtype or "GOAL" in subtype))
                     or "OFFSIDE" in subtype)
            if stops and dead_since is None:
                dead_since = int(event.end)

        if dead_since is not None:
            dead[dead_since:last + 1] = True

        # Kept separate per period so that merging never crosses the half.
        team = owner[first:last + 1].copy()
        team[team < 0] = team[team >= 0][0]
        team = _merge_short_runs(team, dead[first:last + 1])
        labels[first:last + 1] = np.where(dead[first:last + 1], D, np.where(team == 0, H, A))
    return labels, sorted(restarts)


def _merge_short_runs(team, dead):
    """Hand short runs of one team back to the team before them, ignoring dead frames."""
    live = np.flatnonzero(~dead)
    if len(live) == 0:
        return team
    values = team[live].copy()
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(values)]
    for start, stop in zip(starts, stops):
        if start > 0 and stop - start < MIN_POSSESSION_FRAMES:
            values[start:stop] = values[start - 1]
    out = team.copy()
    out[live] = values
    return out


def _slow_noise(rng, shape, sigma=0.3):
    """Position error that drifts over about a second, plus a little jitter.

    Fresh noise on every frame would read as constant player movement and swamp the stillness
    signal, which is what the first version of this did.
    """
    knots = rng.normal(0, sigma, (shape[0] // FPS + 2,) + shape[1:])
    position = np.arange(shape[0]) / FPS
    below = position.astype(int)
    weight = (position - below).reshape((-1,) + (1,) * (len(shape) - 1))
    return knots[below] * (1 - weight) + knots[below + 1] * weight + rng.normal(0, 0.03, shape)


def _degrade(ball, players, team, restarts, n_frames, seed):
    """Make clean sample tracking look like the provided automatic tracking.

    Target figures are the ones DATA.md publishes: ball on 59% of graded frames and about 20.5 of
    22 players tracked. The ball goes missing in bursts and more readily in a crowd, and the
    restart tags are noisy, because every automatic source here is.
    """
    rng = np.random.default_rng(seed)
    ball = ball + _slow_noise(rng, ball.shape)
    players = players + _slow_noise(rng, players.shape)

    distance = np.sqrt(((players - ball[:, None]) ** 2).sum(axis=2))
    with np.errstate(invalid="ignore"):
        crowded = (distance < 4).sum(axis=1) >= 3

    gone = np.zeros(n_frames, dtype=bool)
    frame = 0
    while frame < n_frames:
        visible = int(rng.exponential(9.0 * FPS)) + 1
        hidden = int(rng.exponential(1.6 * FPS)) + 1
        if crowded[min(frame + visible, n_frames - 1)]:
            hidden *= 2
        gone[frame + visible:frame + visible + hidden] = True
        frame += visible + hidden
    ball[gone] = np.nan

    block = 10 * FPS
    for slot in range(players.shape[1]):
        for start in range(0, n_frames, block):
            if rng.random() < 0.07:
                players[start:start + block, slot] = np.nan

    # Identity switches: a track wanders onto a nearby opponent, so its team is wrong for a while.
    home, away = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    for start in range(0, n_frames, block):
        if rng.random() < ID_SWITCH_RATE:
            pair = [rng.choice(home), rng.choice(away)]
            players[start:start + block, pair] = players[start:start + block, pair[::-1]]

    tags = []
    for restart in restarts:
        if rng.random() < 0.6:  # 59 tags against about 100 restarts
            start = restart - 4 * FPS + int(rng.normal(0, 2 * FPS))
            tags.append({"code": "set pieces", "start_frame": start,
                         "end_frame": start + 35 * FPS})
    for _ in range(len(restarts) // 8):  # and invents others
        start = int(rng.integers(0, n_frames))
        tags.append({"code": "set pieces", "start_frame": start, "end_frame": start + 35 * FPS})
    return ball, players, restart_signal(tags, n_frames)


def load_sample(number, seed=0):
    """Sample game `number` (1, 2 or 3), degraded, with labels from its events."""
    loader = _load_epts_game if number == 3 else _load_csv_game
    ball, players, team, events = loader(number)
    ball = ball * PITCH
    players = players * PITCH
    n_frames = len(ball)

    labels, restarts = labels_from_events(events, n_frames)
    ball, players, tags = _degrade(ball, players, team, restarts, n_frames, seed + number)

    graded = np.flatnonzero(labels != X)
    breaks = np.flatnonzero(np.diff(graded) > 1)
    firsts = np.r_[graded[0], graded[breaks + 1]]
    lasts = np.r_[graded[breaks], graded[-1]]
    runs = [(int(a), int(b)) for a, b in zip(firsts, lasts)]
    return Match("sample{}".format(number), ball.astype(np.float32), players.astype(np.float32),
                 team, runs, labels, tags)
