# Possession labelling for More Than a Hack 2027

Our entry to Phase 1 of More Than a Hack 2027. It reads the tracking data for FC Barcelona v
Athletic Club and labels every frame of the tactical video as home possession (`H`), away
possession (`A`) or dead ball (`D`).

The method is a three-state Viterbi decoder over per-frame evidence from player, ball and movement
positions. Its parameters are tuned only on Metrica's open sample games, never on the provided
match. The approach, its limits and the decisions behind it are in
[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md).

The organisers' original README is kept as [README_ORGANISERS.md](README_ORGANISERS.md). Their
starter code, scorer and documentation are MIT licensed by Metrica Sports: see [LICENSE](LICENSE).

## What is ours

| path | what it does |
|---|---|
| `possession/data.py` | loads the match and the sample games, derives sample labels, simulates tracking errors |
| `possession/model.py` | per-frame evidence, Viterbi decoder, minimum stoppage length |
| `possession/tune.py` | parameter search on the sample games, with held-out scoring |
| `possession/select_params.py` | picks between tuned parameter sets on an unseen simulation seed |
| `possession/params.json` | the final parameters |
| `possession/predict.py` | writes the submission |
| `possession/viewer.py` | optional local playback of predictions over the video |

## Regenerate the submission

Tested on Python 3.12.10 on Windows.

**1. Install the pinned dependencies.**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-possession.txt
```

On macOS or Linux the interpreter is `.venv/bin/python`.

**2. Add the match data.** It is gated and is not in this repository. Unpack the bundle into
`data/`, keeping its folder names, or set `MTAH_DATA_DIR`. See [DATA.md](DATA.md).

**3. Write the file.** `TEAM_NAME` is the team name used at registration.

```bash
.venv/Scripts/python -m possession.predict --name TEAM_NAME
```

This reads `possession/params.json` and writes `evaluation/results/TEAM_NAME.csv`. It is
deterministic, so the same data and parameters give a byte-identical file. The first run parses
the tracking data and caches it under `data/`, which takes a few minutes.

**4. Check the file.**

```bash
.venv/Scripts/python evaluation/validate_submission.py evaluation/results/TEAM_NAME.csv
```

## Regenerate the parameters (optional)

This needs Metrica's open sample data next to this repository, at the commit we used:

```bash
git clone https://github.com/metrica-sports/sample-data.git ../sample-data
git -C ../sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
```

Set `MTAH_SAMPLE_DIR` to its `data` folder if you keep it elsewhere. Then tune on two seeds and
select between the results, which overwrites `possession/params.json`:

```bash
.venv/Scripts/python -m possession.tune --seed 0 --no-gaps --off w_tag w_shape --out seed0.json
.venv/Scripts/python -m possession.tune --seed 1 --no-gaps --off w_tag w_shape --out seed1.json
.venv/Scripts/python -m possession.select_params --candidates seed0.json seed1.json
```

## Data and credits

The match data is licensed for this competition only and is not in this repository. `.gitignore`
keeps `data/` and `evaluation/results/*.csv` out of git. Please do not add them.

Sample tracking and event data are from
[Metrica Sports' open sample data](https://github.com/metrica-sports/sample-data).
