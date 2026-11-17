# Decisions

Each choice that changes the labels, with the evidence behind it. Run numbers refer to
[EXPERIMENTS.md](EXPERIMENTS.md).

### D-1: A missing ball is not a dead ball
- **Context:** the ball is missing on 41% of graded frames, mostly during live play. The baseline
  calls every missing-ball frame `D`.
- **Decision:** a missing ball contributes no evidence for `D`. Stoppages come from player movement,
  ball position and the classifier.

### D-2: Decode possessions as intervals, with a 2 s minimum stoppage
- **Context:** spellF1 scores whole possessions, so per-frame flicker is expensive.
- **Decision:** use Viterbi decoding per half with costs for swapping, stopping and restarting, then
  give any dead spell under 2 s to the team before it.
- **Evidence:** without the minimum, a tuned decoder used slivers of `D` as a cheap route between
  teams, producing 150 dead spells under 2 s that split real possessions (run 2).

### D-3: Learned frame evidence instead of hand weights
- **Decision:** a gradient-boosted classifier over many signals at 1, 3 and 8 s scales, with home
  and away mirrored in training so it cannot learn a bias towards one side.
- **Evidence:** run 5 scored 0.358 against 0.271 for v2, better on all 7 matches.

### D-4: Train on IDSSE, not on Metrica
- **Context:** Metrica labels are rebuilt from event data and differ from the challenge's definition.
  The v3 model trained on Metrica found only 8 of 17 possessions on the public window.
- **Decision:** train on the 7 IDSSE matches, whose labels come from the provider's per-frame
  possession and ball status.
- **Evidence:** runs 4 and 5. The IDSSE-trained model also reports 18 possessions on the public
  window, against a true count of 17.

### D-5: Degrade with the ball-gap pattern measured on the provided match
- **Context:** the provided tracking loses the ball in long, rare stretches with a median of 10.7 s.
  Generic simulations, including ours at first and Deen's, used gaps of about 1 s.
- **Decision:** draw lognormal gaps matching the measured median, count and missing share, and add a
  second short-gap copy of each match for robustness.
- **Evidence:** runs 5 to 7. On this bench, a pipeline trained on short gaps scored 0.256.

### D-6: Stop tuning at noise
- **Decision:** reject gains under about 0.01 that do not hold on most matches.
- **Evidence:** runs 8 and 10 gained 0.004 and 0.006. These are the size expected from picking the
  best of many tries on the same folds.

### D-7: Drop Smart Tagging
- **Decision:** do not use Smart Tagging in the final model.
- **Evidence:** it gave no gain on Metrica (run 3), and the IDSSE matches have no equivalent tags,
  so the model could not learn what tags mean.

### D-8: Use a rule-based control model as classifier inputs
- **Decision:** feed teammate Nurazri's deterministic control states (home, away, dead, abstain) to
  the classifier as extra signals, instead of choosing between the two approaches.
- **Evidence:** run 12 scored 0.393 against 0.369, better on 6 of 7 matches. Alone, with his frozen
  v5 thresholds, the rules scored 0.322 on this bench.

## Open questions

- **Long ball gaps:** recovering possession changes inside them, perhaps by estimating the ball's
  path from player movement. Simple features describing the gap did not help (run 9).
- **Restarts:** an explicit restart detector that knows who takes each set piece.
- **Training data:** leagues beyond the Bundesliga.
