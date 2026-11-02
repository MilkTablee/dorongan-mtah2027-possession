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


def features(match):
    """Per-frame arrays, independent of any parameter, so tuning only computes them once."""
    ball, players, team = match.ball, match.players, match.team
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

    window = 3 * FPS
    kernel = np.ones(window) / window
    motion = np.convolve(np.nan_to_num(motion), kernel, mode="same").astype(np.float32)
    # Relative to this match, because tracking sources disagree on absolute speed.
    motion /= np.median(np.concatenate([motion[a:b + 1] for a, b in match.runs]))

    length, width = PITCH
    with np.errstate(invalid="ignore"):
        outside = ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN)
                   | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))

    tags = match.tags if match.tags is not None else np.zeros(len(ball), dtype=np.float32)
    return {"home": near_home, "away": near_away, "ball_speed": ball_speed, "motion": motion,
            "outside": outside, "kickoff": kickoff_shape, "tags": tags}


def emissions(f, p):
    """(T, 3) log-evidence for H, A and D. The two teams are symmetric around zero."""
    with np.errstate(invalid="ignore"):
        closest = np.fmin(f["home"], f["away"])
        controlled = closest < p["radius"]
        margin = np.clip((f["away"] - f["home"]) / p["margin"], -1, 1)
        fast = f["ball_speed"] > p["fast"]
    team = np.where(controlled, margin, 0.0) * p["w_team"]
    team = np.where(fast, team * p["flight"], team)
    team = np.nan_to_num(team)

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


def predict(match, f, p):
    """Letters for every frame of the match, X outside the runs."""
    out = np.full(len(match.ball), "X", dtype="<U1")
    score = emissions(f, p)
    for first, last in match.runs:
        out[first:last + 1] = LETTERS[viterbi(score[first:last + 1], p)]
    return out
