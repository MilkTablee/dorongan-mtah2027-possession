# Possession labelling for More Than a Hack 2027 (team Dorongan)

Our entry to Phase 1 of More Than a Hack 2027. It reads the tracking data for FC Barcelona v
Athletic Club and labels every frame of the tactical video as home possession (`H`), away
possession (`A`) or dead ball (`D`).

The method is a gradient-boosted frame classifier feeding a three-state Viterbi decoder. It is
trained on seven open Bundesliga matches (IDSSE), never on the provided match. The approach, its
limits and the decisions behind it are in [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md).

## Results

spellF1, leaving out one of 7 held-out Bundesliga matches at a time. The tracking is degraded with
the ball-gap pattern measured on the provided match:

| model | mean | per match |
|---|---|---|
| Nearest player, ±50 frame vote (the published floor's algorithm) | 0.178 | 0.168–0.188 |
| v2: hand-weighted evidence | 0.271 | 0.256–0.294 |
| Learned evidence, IDSSE training | 0.369 | 0.322–0.419 |
| **Submitted: + rule-based control states as inputs** | **0.393** | **0.354–0.446** |

The submitted model is better than v2 on every match. The full history, including what failed, is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The reasons for each choice are in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Tests

```bash
.venv/Scripts/python -m pip install pytest
.venv/Scripts/python -m pytest tests
```

These are fast unit tests for the decoder and the classifier's inputs, and they need no match
data.

The organisers' original README is kept as [README_ORGANISERS.md](README_ORGANISERS.md). Their
starter code, scorer and documentation are MIT licensed by Metrica Sports: see [LICENSE](LICENSE).

## What is ours

| path | what it does |
|---|---|
| `possession/learned.py` | **the submitted model:** features, classifier, decoder, evaluation, submission |
| `possession/rules_control.py` | teammate Nurazri's rule-based control model, used as classifier inputs |
| `possession/external.py` | loads the IDSSE matches and degrades them to look like the provided tracking |
| `possession/model.py` | per-frame signals, Viterbi decoder, minimum stoppage length |
| `possession/data.py` | loads the provided match and the Metrica samples, simulates tracking errors |
| `possession/learned_params.json` | the decoder costs used for the submission |
| `possession/idsse_sha256.txt` | checksums of the 14 IDSSE files we trained on |
| `possession/tune.py`, `select_params.py`, `predict.py`, `params.json` | the earlier hand-tuned model (v2), kept for comparison |
| `possession/viewer.py` | optional local playback of predictions over the video |

## Regenerate the submission

Tested on Python 3.12.10 on Windows. The full run takes about 10 minutes.

**1. Install the pinned dependencies.**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-possession.txt
```

On macOS or Linux the interpreter is `.venv/bin/python`.

**2. Add the match data.** It is gated and is not in this repository. Unpack the bundle into
`data/`, keeping its folder names, or set `MTAH_DATA_DIR`. See [DATA.md](DATA.md).

**3. Download the training data.** These are the 7 IDSSE matches from Bassek et al. (2025),
published on figshare under CC-BY. Each match has a match information file and a positions file,
14 files in all, about 2.6 GB. Download them in a browser, because figshare does not serve
scripted downloads:

| match | match information | positions |
|---|---|---|
| J03WMX | [51643475](https://springernature.figshare.com/ndownloader/files/51643475) | [51643514](https://springernature.figshare.com/ndownloader/files/51643514) |
| J03WN1 | [51643472](https://springernature.figshare.com/ndownloader/files/51643472) | [51643517](https://springernature.figshare.com/ndownloader/files/51643517) |
| J03WOH | [51643478](https://springernature.figshare.com/ndownloader/files/51643478) | [51643520](https://springernature.figshare.com/ndownloader/files/51643520) |
| J03WOY | [51643481](https://springernature.figshare.com/ndownloader/files/51643481) | [51643523](https://springernature.figshare.com/ndownloader/files/51643523) |
| J03WPY | [51643487](https://springernature.figshare.com/ndownloader/files/51643487) | [51643526](https://springernature.figshare.com/ndownloader/files/51643526) |
| J03WQQ | [51643484](https://springernature.figshare.com/ndownloader/files/51643484) | [51643529](https://springernature.figshare.com/ndownloader/files/51643529) |
| J03WR9 | [51643490](https://springernature.figshare.com/ndownloader/files/51643490) | [51643532](https://springernature.figshare.com/ndownloader/files/51643532) |

Save them, with the names they arrive with, in a folder named `idsse` next to this repository,
or set `MTAH_IDSSE_DIR`. Check them against `possession/idsse_sha256.txt`: a different file
gives a different submission.

**4. Write the file.** `TEAM_NAME` is the team name used at registration.

```bash
.venv/Scripts/python -m possession.learned --source idsse --augment mixed --name TEAM_NAME
```

This writes `evaluation/results/TEAM_NAME.csv` and `possession/learned_params.json`. Every random
step is seeded, so repeated runs give a byte-identical file. The first run parses and caches the
IDSSE files and the match tracking, which takes a few extra minutes.

**5. Check the file.**

```bash
.venv/Scripts/python evaluation/validate_submission.py evaluation/results/TEAM_NAME.csv
```

**To reproduce the evaluation** reported in the technical document, run this. It does leave-one-match-out over the 7 IDSSE
matches, against the floor and the earlier model:

```bash
.venv/Scripts/python -m possession.learned --evaluate --source idsse --augment mixed --seed 0
```

## Data and credits

The match data is licensed for this competition only and is not in this repository. `.gitignore`
keeps `data/` and `evaluation/results/*.csv` out of git. Please do not add them.

- Training data: Bassek, M., et al. (2025). *An integrated dataset of spatiotemporal and event data
  in elite soccer.* Scientific Data. IDSSE dataset, CC-BY 4.0.
- Development data: [Metrica Sports' open sample data](https://github.com/metrica-sports/sample-data).
- The IDSSE file list and checksums come from teammate Deen's exploratory repository,
  [DeenSkuad/mtah-2027-dorongan](https://github.com/DeenSkuad/mtah-2027-dorongan).
