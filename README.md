<h1 align="center">奈奈の掉落预测 · Model</h1>

<p align="center">
  Python package for training, offline inference, serving, and model comparison behind the WoW loot predictor.
</p>

This repo powers the **model layer** for the loot prediction flow. Given a `raidId`, `instanceId`, and boss loot pools, it predicts which items each boss will drop.

## Model boundary

- `LootModel.predict` predicts loot from `raidId`, `instanceId`, and boss loot pools.
- `TrainableLootModel.train` defines the training interface for observed loot runs.
- `DeterministicLootModel` is the seeded baseline equivalent to the current backend mock.

Future CatBoost/sklearn classifiers, period-residue baselines, data normalization, and accuracy comparison runners should live here instead of inside the Nest API.

## Local commands

```bash
python -m sea_of_blossoms_model.cli.predict < prediction-input.json
python -m sea_of_blossoms_model.cli.train --input observed-runs.json
```

## Sample E2E

Generate observed training data from the current deterministic baseline:

```bash
PYTHONPATH=src python -m sea_of_blossoms_model.cli.prepare_training_data \
  --catalog /path/to/raids.json \
  --output data/sample-observed-runs.json \
  --raid-id 13 \
  --instance-start 12330000 \
  --instance-count 64
```

Train a residue-frequency artifact:

```bash
PYTHONPATH=src python -m sea_of_blossoms_model.cli.train \
  --input data/sample-observed-runs.json \
  --artifact artifacts/current-model.json \
  --metrics artifacts/sample-residue-metrics.json \
  --model residue-frequency \
  --period 16
```

Run prediction with the artifact:

```bash
PYTHONPATH=src python -m sea_of_blossoms_model.cli.predict \
  --input /path/to/prediction-input.json \
  --artifact artifacts/current-model.json
```

Run the local model server:

```bash
PYTHONPATH=src python -m sea_of_blossoms_model.cli.serve \
  --artifact artifacts/current-model.json \
  --host 127.0.0.1 \
  --port 4317
```

Point the API at the local server:

```bash
LOOT_PREDICTOR_STRATEGY=model \
MODEL_ENDPOINT_URL=http://127.0.0.1:4317/predict \
PORT=3100 \
npm run start:dev
```

(Run from the backend API project.)

## Hosting

The infrastructure project hosts this package as a Python Lambda container image for the first production-shaped serving path. The Lambda handler is `sea_of_blossoms_model.lambda_handler.handler`.

Generated model artifacts are ignored by Git. Build or provide `artifacts/current-model.json` before creating an artifact-backed image. If `MODEL_ARTIFACT_PATH` is unset or points to a missing file, the handler falls back to the deterministic baseline.

The backend should call a configured model runtime or consume exported artifacts. It should not own training logic.
