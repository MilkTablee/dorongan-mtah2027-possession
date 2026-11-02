"""Frame evidence and the three-state decoder.

Every frame gives evidence for H, A and D, and Viterbi picks the best label sequence given a cost
for each change of state. Possession is an interval, so the costs do the work the baseline's
majority vote was doing, without blurring the edges of a possession.

The one rule that matters most: a missing ball is no evidence at all. It is not a dead ball.
"""

import numpy as np

FPS = 25
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
PITCH = (105.0, 68.0)
OUT_MARGIN = 0.5      # metres past a line before the ball counts as out
KICKOFF_MARGIN = 1.0  # metres of overlap tolerated between the two teams at a kickoff
GAP_FRAMES = FPS * 3 // 2  # ball gaps up to this long are bridged in a straight line
MIN_DEAD_FRAMES = 2 * FPS  # a stoppage shorter than this is a decoding artefact


def fill_ball_gaps(ball, limit=GAP_FRAMES):
    """Bridge short gaps linearly. A ball rarely turns far inside 1.5 s, and gaps are the rule."""
    out = ball.copy()
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    frames = np.arange(len(ball))
    for axis in (0, 1):
        out[:, axis] = np.interp(frames, seen, ball[seen, axis])
    after = np.searchsorted(seen, frames).clip(1, len(seen) - 1)
    out[seen[after] - seen[after - 1] > limit] = np.nan
    out[~np.isnan(ball[:, 0])] = ball[~np.isnan(ball[:, 0])]
    return out


def features(match, gap=GAP_FRAMES):
    """Per-frame arrays, independent of any parameter, so tuning only computes them once."""
    ball, players, team = fill_ball_gaps(match.ball, gap), match.players, match.team
    away = team == 1

    distance = np.sqrt(((players - ball[:, None]) ** 2).sum(axis=2))
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames stay NaN
            near_home = np.nanmin(distance[:, ~away], axis=1)
            near_away = np.nanmin(distance[:, away], axis=1)

            # Over 0.2 s, to keep tracking jitter out of the speed.
            step = 5
            ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
            ball_speed[step:] = np.sqrt(((ball[step:] - ball[:-step]) ** 2).sum(axis=1)) * FPS / step

            moved = np.sqrt(((players[step:] - players[:-step]) ** 2).sum(axis=2)) * FPS / step
            motion = np.zeros(len(ball), dtype=np.float32)
            motion[step:] = np.nan_to_num(np.nanmean(moved, axis=1))

            home_x = np.nanmean(players[:, ~away, 0], axis=1)
            away_x = np.nanmean(players[:, away, 0], axis=1)
            home_left = home_x < away_x
            left_edge = np.where(home_left, np.nanmax(players[:, ~away, 0], axis=1),
                                 np.nanmax(players[:, away, 0], axis=1))
            right_edge = np.where(home_left, np.nanmin(players[:, away, 0], axis=1),
                                  np.nanmin(players[:, ~away, 0], axis=1))
            kickoff_shape = left_edge < right_edge + KICKOFF_MARGIN

            # The side with the ball spreads out and the other side compacts, ball seen or not.
            spread = (np.sqrt(np.nanvar(players[:, ~away, 0], axis=1)
                              + np.nanvar(players[:, ~away, 1], axis=1))
                      - np.sqrt(np.nanvar(players[:, away, 0], axis=1)
                                + np.nanvar(players[:, away, 1], axis=1)))

    window = 3 * FPS
    kernel = np.ones(window) / window
    motion = np.convolve(np.nan_to_num(motion), kernel, mode="same").astype(np.float32)
    # Relative to this match, because tracking sources disagree on absolute speed.
    motion /= np.median(np.concatenate([motion[a:b + 1] for a, b in match.runs]))

    spread = np.convolve(np.nan_to_num(spread), kernel, mode="same").astype(np.float32)
    spread -= np.median(spread)
    spread /= max(float(np.percentile(np.abs(spread), 90)), 1e-6)

    length, width = PITCH
    with np.errstate(invalid="ignore"):
        outside = ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN)
                   | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))

    tags = match.tags if match.tags is not None else np.zeros(len(ball), dtype=np.float32)
    return {"home": near_home, "away": near_away, "ball_speed": ball_speed, "motion": motion,
            "outside": outside, "kickoff": kickoff_shape, "tags": tags,
            "spread": spread}


def emissions(f, p):
    """(T, 3) log-evidence for H, A and D. The two teams are symmetric around zero."""
    with np.errstate(invalid="ignore"):
        closest = np.fmin(f["home"], f["away"])
        controlled = closest < p["radius"]
        margin = np.clip((f["away"] - f["home"]) / p["margin"], -1, 1)
        fast = f["ball_speed"] > p["fast"]
    team = np.where(controlled, margin, 0.0) * p["w_team"]
    team = np.where(fast, team * p["flight"], team)
    team = np.nan_to_num(team) + p.get("w_shape", 0.0) * np.clip(f["spread"], -1, 1)

    stillness = np.clip((p["still"] - f["motion"]) / p["still"], -1, 1)
    dead = (p["w_still"] * stillness - p["bias"]
            + p["w_out"] * f["outside"]
            + p["w_kickoff"] * f["kickoff"]
            + p["w_tag"] * f["tags"]
            - p["w_fast"] * np.nan_to_num(fast))
    return np.stack([team, -team, dead], axis=1)


def viterbi(score, p):
    """Best label sequence for one half. Plain Python lists: fast enough, and easy to read."""
    swap, stop, start = p["swap"], p["stop"], p["start"]
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    rows = score.tolist()
    h, a, d = rows[0]

    for i in range(1, n):
        eh, ea, ed = rows[i]
        to_h, from_h = max((h, 0), (a - swap, 1), (d - start, 2))
        to_a, from_a = max((a, 1), (h - swap, 0), (d - start, 2))
        to_d, from_d = max((d, 2), (h - stop, 0), (a - stop, 1))
        back[i, 0], back[i, 1], back[i, 2] = from_h, from_a, from_d
        h, a, d = to_h + eh, to_a + ea, to_d + ed

    state = int(np.argmax([h, a, d]))
    path = np.empty(n, dtype=np.int8)
    for i in range(n - 1, -1, -1):
        path[i] = state
        state = back[i, state]
    return path


def absorb_short_dead(path):
    """Give dead spells too short to be stoppages to the team before them.

    Without this the decoder can hop H -> D -> A through a sliver of D when a direct swap is
    expensive, and each sliver splits a real possession in two.
    """
    out = path.copy()
    edges = np.flatnonzero(np.diff(path)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(path)]
    for start, stop in zip(starts, stops):
        if path[start] == D and stop - start < MIN_DEAD_FRAMES and start > 0:
            out[start:stop] = out[start - 1]
    return out


def predict(match, f, p):
    """Letters for every frame of the match, X outside the runs."""
    out = np.full(len(match.ball), "X", dtype="<U1")
    score = emissions(f, p)
    for first, last in match.runs:
        out[first:last + 1] = LETTERS[absorb_short_dead(viterbi(score[first:last + 1], p))]
    return out
