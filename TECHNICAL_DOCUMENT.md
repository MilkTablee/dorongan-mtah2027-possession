# Technical document: learned frame evidence with three-state decoding

More Than a Hack 2027, Phase 1. Team Dorongan. FC Barcelona v Athletic Club.

## 1. Approach

We treat possession as a segmentation problem. The ranking metric, spellF1, scores whole possessions
rather than single frames, so one false change of state inside a possession can cost more than many
wrong frames at its edges. The system has three stages:

1. **Classify.** A gradient-boosted classifier (scikit-learn `HistGradientBoostingClassifier`) gives
   every frame a probability of home possession (`H`), away possession (`A`) and dead ball (`D`).
2. **Decode.** A Viterbi decoder finds the most likely label sequence for each half from those
   probabilities, with a fixed cost for each change of state: 10 for a swap between teams and 24
   for stopping or restarting play.
3. **Clean up.** Any dead-ball spell under 2 s is given to the team before it, because no real
   stoppage is that short. An earlier version used such slivers as a cheap route between teams,
   which split real possessions.

**Why this design.** The nearest-player baseline treats a missing ball as a dead ball, although the
ball is missing on 41% of graded frames, and it has no notion of a possession as an interval. We
first built a hand-weighted version of this decoder (v2 below). Replacing its hand-set evidence
with a classifier that sees many signals at several time scales was the largest single gain we
measured.

### What the classifier sees

Every frame has a set of base signals. Each signal is used both raw and as a centred 1 s, 3 s and
8 s average. Each average also comes with the share of that window where the signal existed, so the
model knows when it is working from little evidence.

**Signals that do not depend on team:**
- ball speed
- mean player speed, divided by the match median so that tracking sources are comparable
- ball out of play, and ball distance to the nearest line and to the nearest goal line
- whether the ball is seen
- the distance from the ball to the closest player
- a kickoff shape, where the two teams are separated across a vertical line

**Signals that depend on team:**
- the difference between each team's nearest player to the ball
- the difference in how spread out the two teams are

**Rule-based control states as extra inputs.** Teammate Nurazri's deterministic control model
(`possession/rules_control.py`, from [nurazri/dorongan-hackathon](https://github.com/nurazri/dorongan-hackathon))
is run on the same tracking with the thresholds written in that file (his earlier v2 values, without
Smart Tagging). Its per-frame state (home control, away
control, dead ball or abstain) becomes three more signals: dead, abstain, and a signed team state.
Its sharp rules for a settled ball far from players, and for a restart near a player, complement the
classifier's averaged signals.

**Home and away are mirrored in training.** Every training frame appears twice, once with the
teams swapped. At prediction time we average both orientations. The model therefore cannot learn
that one side has the ball more often, which would carry from the training matches to this one.

## 2. Assets used and dropped

| asset | status |
|---|---|
| Tactical tracking (ATD) | **Used.** The only input at prediction time |
| Smart Tagging | **Dropped.** The training matches have no equivalent tags, and an earlier test showed no gain |
| Homography | Not used. We work in tracking coordinates |
| Tactical video | Not an input. Used only for inspection tooling |
| Broadcast video and tracking | Not used. Not synchronised to the tactical timeline |

## 3. Training data

**No labels from the provided match were used for training, tuning or validation, and nobody
labelled any data by hand.**

**Training data: 7 IDSSE matches.** These are Bundesliga matches from Bassek et al. (2025), *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, CC-BY 4.0.
Together they have about 1 million frames. Their tracking carries per-frame ball possession and
ball status produced by the data provider, DFL, which is the same kind of label as the challenge
ground truth. Our labels are `D` whenever the ball status is dead, otherwise the last team recorded
in possession, reset at each half.

**The IDSSE tracking is degraded to look like the provided automatic tracking.** IDSSE tracking
is corrected and nearly complete, and the provided tracking is neither. We add:
- slowly drifting position error
- player tracks that drop out for 10 s at a time
- identity switches, where a home track and an away track swap for 10 s
- ball gaps copying how the ball goes missing in the provided match

For the gaps, we measured three summary statistics on the provided tracking, without using any
labels: 40.7% of graded frames missing, about 150 gaps, and a median gap of 10.7 s. We then draw
lognormal gaps to match. Each match is also used a second time with short, frequent ball gaps
instead, so the model does not depend on our gap measurement being right. This matters because the ball in the provided match goes missing in long,
rare stretches, not short flickers. After degradation the nearest-player baseline scores 0.178 on
these matches, against 0.2433 on the real one, so our bench is slightly harder than the real task.

**Development data: Metrica's open sample games.** [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
was used to develop and tune the earlier v1 and v2 models. For those games our code derives frame
labels from Metrica's human-produced event data.

**Credit.** The idea of training on IDSSE, the file list and the checksums came from teammate Deen's
exploratory repository, [DeenSkuad/mtah-2027-dorongan](https://github.com/DeenSkuad/mtah-2027-dorongan).

### Model selection

We used leave-one-match-out over the 7 IDSSE matches. Each match is scored by a model trained on
the other six. The decoder costs were picked from a 5 × 5 grid on those same out-of-fold
predictions. That is a choice of two numbers made on the reported predictions, so the learned
figures below are slightly optimistic. The final model is trained on all seven matches.

### Use of the public ground truth

We used the 5-minute public window only to check that the pipeline was correct. It played no part
in training, tuning or selecting the model.

## 4. Possession rules implemented

- **Loose balls:** possession stays with the current team unless the evidence for the other team
  builds up enough to pay the swap cost.
- **Challenges and deflections:** a brief touch by the other team does not pay the swap cost. There
  is no explicit challenge detector. The classifier learns from the provider's labels what counts
  as a change of possession.
- **Restarts:** after a stoppage, the decoder picks the team with the stronger evidence. One input
  is the rule model's restart trigger: an in-bounds ball moving at 1.5 m/s or more within 4 m of a player.
- **Stoppages:** a stoppage needs enough accumulated `D` probability and must last at least 2 s. Inputs
  include the rule model's hand rules: a ball out of play, or a settled ball at least 3 m from any player.
- **Second-half pre-kickoff:** handled by the same evidence. No frame numbers from the match are
  written into the model.

## 5. Results

**spellF1, leave-one-match-out on 7 degraded IDSSE matches:**

| model | mean | per match |
|---|---|---|
| Nearest player, ±50 frame vote (the published floor's algorithm) | 0.178 | 0.168–0.188 |
| v2: hand-weighted evidence, Viterbi decoder | 0.271 | 0.256–0.294 |
| v3: learned evidence, Viterbi decoder | 0.358 | 0.314–0.414 |
| v3 + second copy, different seed | 0.362 | 0.318–0.427 |
| v3 + second copy with short ball gaps | 0.369 | 0.322–0.419 |
| **+ rule-based control states as inputs (submitted)** | **0.393** | **0.354–0.446** |

v3 beats v2 on every one of the 7 matches, by 0.05 to 0.14. The organisers give about 0.04 as the
noise band for spellF1 on a full match. Adding the short-gap copy gains 0.011, which is below that
band, but it is better or equal on all 7 matches when both versions are scored on the same
matches. We kept it as a guard against our measured gap pattern being incomplete. Adding the rule-based
control states gained a further 0.024, better on 6 of 7 matches by about 0.03 each.

**Public 5 minutes (17 possessions), as a sanity check only:** spellF1 0.500 and macroF1 0.699. The
model reports 15 possessions, against a true count of 17.

**Full match:** the model reports 39.1% `H`, 26.3% `A` and 34.6% `D`, with 301 possessions averaging
13.8 s, and 121 stoppages. The private full-match score is unknown.

## 6. Limitations

- **The degradation is a model of the real tracking errors, not a copy of them.** Bench scores will
  differ from the real ones. Ours is harder than the real task by the floor's measure, so we expect
  real scores to be somewhat higher, but we cannot verify that.
- **Long ball gaps are the hardest case.** When the ball is missing for 10 s or more, the model works
  only from player movement and shape, and changes of possession in that time are found late or not
  at all.
- **Label meaning may differ slightly.** IDSSE labels come from DFL's possession and ball-status
  definitions, which may not match exactly what the challenge's corrected tracking means by "dead"
  and "owning".
- **Seven training matches from one league.** That is enough to beat our hand model consistently
  but small for a learned model.
- **Decoder costs were chosen on the reported folds,** so the bench figures are slightly optimistic.
- **There is no identity handling.** Track slots swap identities in the real data, and only the
  degradation simulates this. The model does not correct it.

With more time we would estimate ball position across gaps from player movement, add an explicit
restart detector that knows who takes each set piece, and train on more leagues.

## 7. Reproduction

[README.md](README.md) gives the full steps. In short, on Python 3.12.10 with
`requirements-possession.txt` installed, the match data in `data/` and the 14 IDSSE files in
`../idsse`:

```bash
python -m possession.learned --source idsse --augment mixed --name Dorongan
```

This trains the model and writes `evaluation/results/Dorongan.csv`. Every random step is seeded,
and two runs produced byte-identical files.
