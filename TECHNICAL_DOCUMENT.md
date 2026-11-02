# Technical document: three-state possession decoding

More Than a Hack 2027, Phase 1. FC Barcelona v Athletic Club.

## 1. Approach

We treat possession as a segmentation problem. The ranking metric, spellF1, scores whole possessions
rather than single frames, so one false change of state inside a possession can cost more than many
wrong frames at its edges.

Each frame gets a score for each of three states: home possession (`H`), away possession (`A`) and
dead ball (`D`). A Viterbi decoder then finds the best label sequence for each half, given a fixed
cost for each change of state. Those costs do the job of the baseline's majority vote, without
blurring the boundaries of a possession.

**Why this rather than something else.** The nearest-player baseline makes two mistakes we wanted to
remove. It treats a missing ball as a dead ball, although the ball is missing on 41% of graded frames
and mostly during live play. It also has no notion of a possession as an interval. We chose a small,
inspectable model over a large learned one because we have one match to label and no permitted
labels on it, so anything with many parameters would be hard to validate.

### Evidence for each frame

**Team evidence** is the difference between each team's nearest player to the ball, in metres. It
counts only when a player is within a control radius of the ball, and it is discounted while the ball
is moving fast, because a ball passing a player is not a ball under control. If the ball is missing,
or nobody is close, the team evidence is zero. This matches the challenge rule that possession stays
with the last team to have it during loose or contested phases.

**Dead-ball evidence** combines four signals:

- **Stillness:** mean player speed over 3 s, divided by the match median. Scaling by the match makes
  it comparable across tracking sources. This is the strongest signal we found. On the sample games,
  relative movement is about 0.65 at a stoppage and about 1.2 in live play.
- **Ball out of play:** the ball is more than 0.5 m past a line.
- **Kickoff shape:** the two teams are fully separated across a vertical line.
- **Fast ball:** a ball moving fast inside the pitch counts against a stoppage.

A missing ball gives no evidence towards `D`.

### Decoding

Decoding runs separately on each half, with three costs: swapping team, stopping play and restarting
play. After decoding, any dead-ball spell under 2 s is given to the team before it. We added this
after a tuned model used slivers of `D` as a cheap route between teams, which split real possessions.

## 2. Assets used and dropped

| asset | status |
|---|---|
| Tactical tracking (ATD) | **Used.** The only input to the final model |
| Smart Tagging | **Tried and dropped.** No measurable gain |
| Homography | Not used. We work in tracking coordinates |
| Tactical video | Not an input. Used only for inspection tooling |
| Broadcast video and tracking | Not used. Not synchronised to the tactical timeline |

We also tried two additions that are switched off in the final model. The code remains in the
repository so that these results can be checked.

| variant, with the 2 s minimum stoppage | held-out spellF1, seed 0 | seed 1 |
|---|---|---|
| **Final model: no additions** | **0.468** | **0.513** |
| Ball-gap bridging + Smart Tagging + team spread | 0.464 | 0.476 |
| Ball-gap bridging + Smart Tagging | 0.437 | 0.451 |

Ball-gap bridging fills ball gaps of up to 1.5 s in a straight line. Team spread uses the idea that
the side with the ball spreads out. For Smart Tagging, mean player movement around tag starts showed
that set-piece tags begin a few seconds before the restart, so only the tag start was used for
timing. That measurement used movement only and no labels.

These gaps are about the size of the variation between seeds, so they do not show that the additions
harm the model. They show that the additions did not earn their extra complexity.

## 3. Training data

No labels from the provided match were used to train, tune or validate the model, and nobody labelled
any data by hand for this entry.

All tuning used the **three open sample games** published by Metrica Sports, which are other matches:
[metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), commit `e706dd5`. Together
they give about 425,000 labelled frames and 851 possessions.

Those games ship with human-produced event data, created by Metrica. We did not label anything
ourselves. Our code derives frame labels from those events (`labels_from_events` in
`possession/data.py`):

- The ball dies on a ball out, a shot that leaves the pitch or scores, a foul or an offside.
- It lives again at the next restart.
- The owner is the team that last made a controlled action.
- Hand-overs under 1.5 s are merged back, because a touch is not a possession.

Because the sample tracking is much cleaner than the provided automatic
tracking, we degrade it before tuning. The simulation hides the ball in bursts, more often in crowds;
adds slowly drifting position error; removes player tracks for short periods; and swaps identities
between home and away tracks. The published coverage figures guided these settings. The simulation
stays optimistic: the nearest-player algorithm scores about 0.26–0.35 on it, against 0.2433 on the
real match.

### Tuning

We ran a random search of 300 settings, then 150 more around the best one, scored with the official
spellF1. Each game was held out in turn: settings were chosen on two games and scored on the third.
We repeated this on two simulation seeds, then chose between the two resulting parameter sets on a
third seed that neither had been tuned on.

### Use of the public ground truth

We used the 5-minute public window only to check that the pipeline was correct. It played no part in
tuning or in selecting the final model. Our final model scores lower there than our first version,
and we kept it because the public window cannot rank submissions.

## 4. Possession rules implemented

- **Loose balls:** with no player in control range there is no team evidence, so the current team
  keeps possession unless the evidence builds against it.
- **Challenges and deflections:** a brief touch by the other team is not enough on its own to pay the
  cost of a team swap. There is no explicit challenge detector.
- **Restarts:** after a stoppage, the decoder picks the team with the stronger evidence. It does not
  use a restart event or infer who will take the set piece.
- **Stoppages:** there is no fixed rule. A stoppage needs enough accumulated evidence, mostly
  stillness, and must last at least 2 s.
- **Second-half pre-kickoff:** these frames are handled by the same dead-ball evidence. No frame
  numbers from the match are written into the model.

## 5. Results

| measure | spellF1 | macroF1 |
|---|---|---|
| Sample games, held out, mean of two seeds | 0.490 | not measured |
| Sample games, unseen seeds 2 to 4, final settings | 0.472–0.476 | not measured |
| Public 5 minutes (17 possessions) | 0.463 (90% interval 0.30–0.64) | 0.707 |

The score on the private full match is unknown. We expect it to be lower than the sample figures,
because the sample simulation is easier than the real tracking. The seed 2 to 4 runs reuse the same
three games, so they test different noise rather than new matches, and those settings were tuned on
those games.

Across the graded match the model reports 34.1% `H`, 20.5% `A` and 45.4% `D`, with 268 possessions of
about 13 s on average and 139 stoppages.

## 6. Limitations

- **Dead-ball detection is the weakest part.** In the public window, the model finds only 36% of true
  `D` frames, with 66% precision. Short stoppages with little player movement change are likely to be
  missed.
- **Possession edges are imprecise.** The team swap cost trades false swaps for late or missed ones.
  Latency in the public window is about 8 frames early.
- **Stretches without the ball are nearly blind.** With the ball missing, the model mostly holds its
  current state, so a change of possession during a long gap is found late or not at all.
- **Tuning labels are approximate.** They come from event data, not the corrected tracking behind the
  real ground truth, so "dead" and "owning" may differ slightly in meaning.
- **Three sample games is a small validation set,** and differences of about 0.04 are noise.
- **No identity handling.** Track slots swap identities in the real data, and the model does not try
  to correct this.

With more time we would build a better restart detector, add evidence for who takes each restart,
model ball trajectories across gaps with player context rather than in a straight line, and improve
stoppage detection using the video.

## 7. Reproduction

[README.md](README.md) gives the full steps. In short, on Python 3.12.10 with
`requirements-possession.txt` installed and the match data in `data/`:

```bash
python -m possession.predict --name TEAM_NAME
```

This reads `possession/params.json` and writes `evaluation/results/TEAM_NAME.csv`. Two runs produced
byte-identical files. The README also gives the commands that regenerate `params.json` from the
sample games.
