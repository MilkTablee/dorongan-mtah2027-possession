# Experiment log

Every experiment that shaped the submission, including the ones that failed. All scores are
spellF1, the ranking metric. The public 5-minute window was used only to check the pipeline,
never to choose between versions.

## Benches

| bench | matches | labels | degradation | floor (nearest player, ±50 vote) |
|---|---|---|---|---|
| **IDSSE** (decides from v3 on) | 7 Bundesliga, leave one out | provider's per-frame possession and ball status | ball gaps measured on the provided tracking, plus noise, dropout and identity switches | **0.178** |
| Metrica | 3 open sample games | rebuilt from event data | short ball gaps, noise, dropout and identity switches | 0.26–0.35 |

The floor's published score on the real match is 0.2433, so the IDSSE bench is slightly harder than
the real task and the Metrica bench is easier.

**Measured on the provided tracking, without labels:** the ball is missing on 40.7% of graded frames,
in about 150 gaps with a median length of 10.7 s, and two thirds of the missing frames fall in
gaps over 20 s. The ball goes missing in long, rare stretches, not short flickers, so the IDSSE
degradation copies this pattern.

## Runs

| # | model | bench | spellF1 | notes |
|---|---|---|---|---|
| 1 | v1: hand-weighted evidence, Viterbi | Metrica | 0.56 | the simulation added per-frame noise, which faked constant player movement; fixed in run 2 |
| 2 | v2: as v1, tuned on harder simulation, minimum 2 s stoppage | Metrica | 0.47–0.48 | beat v1 by about 0.04 on 3 unseen noise seeds |
| 3 | v2 + ball-gap bridging, tags, team spread | Metrica | 0.44–0.47 | no gain; switched off |
| 4 | v3: learned classifier, Viterbi | Metrica | 0.52–0.53 | +0.04 over v2, but only 8 of 17 possessions on the public window; dropped |
| 5 | **v3: learned classifier, trained on IDSSE** | IDSSE | **0.358** | v2 0.271 and floor 0.178 on the same matches; beats v2 on all 7 |
| 6 | v3 + second copy, different seed | IDSSE | 0.362 | better on 5 of 7 |
| 7 | **v3 + second copy with short ball gaps (submitted)** | IDSSE | **0.369** | better or equal on all 7; kept as a guard against our gap measurement |
| 8 | run 7 with a wider decoder-cost grid (175 settings) | IDSSE | 0.373 at best | the best of 175 on the same folds, better on 5 of 7; treated as noise |
| 9 | run 7 + long-gap features (time since ball seen, last position, 20 s window) | IDSSE | 0.356 | worse on 5 of 7 |
| 10 | run 7 with 63 leaves instead of 31 | IDSSE | 0.374 | +0.006 with different decoder costs; noise |
| 11 | runs 9 + 10 | IDSSE | 0.360 | worse |

## Per match, IDSSE bench

| match | floor | v2 | v3 | **submitted (run 7)** |
|---|---|---|---|---|
| J03WMX | 0.188 | 0.264 | 0.328 | **0.351** |
| J03WN1 | 0.178 | 0.256 | 0.314 | **0.322** |
| J03WOH | 0.179 | 0.277 | 0.379 | **0.396** |
| J03WOY | 0.169 | 0.294 | 0.360 | **0.376** |
| J03WPY | 0.168 | 0.259 | 0.346 | **0.354** |
| J03WQQ | 0.182 | 0.271 | 0.414 | **0.419** |
| J03WR9 | 0.179 | 0.278 | 0.363 | **0.363** |
| **mean** | **0.178** | **0.271** | **0.358** | **0.369** |

## Comparison with a teammate's pipeline

Teammate Deen built a similar pipeline separately, in
[DeenSkuad/mtah-2027-dorongan](https://github.com/DeenSkuad/mtah-2027-dorongan). We trained it
exactly as his code does and scored it on our IDSSE bench, leaving one match out:

| | mean spellF1 |
|---|---|
| Deen's pipeline | 0.256 |
| **Ours (submitted)** | **0.369**, higher on all 7 matches |

**Caveats.** This run used his default degradation settings, because his locally measured settings
are not in his repository. His model was also trained on short ball gaps, while this bench uses the
long gaps measured on the provided match. The comparison therefore mostly measures how well each
pipeline copes with long ball gaps, which is the condition the real match has.

## Public 5 minutes, sanity only

| version | spellF1 | possessions (true: 17) |
|---|---|---|
| baseline_smooth50 | 0.326 | 26 |
| v2 | 0.463 | 15 |
| **submitted** | **0.457** | **18** |
