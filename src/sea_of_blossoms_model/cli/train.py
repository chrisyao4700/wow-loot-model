from __future__ import annotations

import argparse
import json
from pathlib import Path

from sea_of_blossoms_model.baselines import DeterministicLootModel
from sea_of_blossoms_model.models import ResidueFrequencyLootModel
from sea_of_blossoms_model.serialization import training_input_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or register a loot model.")
    parser.add_argument("--input", type=Path, required=True, help="Observed training data JSON.")
    parser.add_argument("--artifact", type=Path, help="Where to write a trained model artifact.")
    parser.add_argument("--metrics", type=Path, help="Where to write training metrics JSON.")
    parser.add_argument(
        "--model",
        choices=("residue-frequency", "deterministic"),
        default="residue-frequency",
        help="Model implementation to train.",
    )
    parser.add_argument("--period", type=int, default=97, help="instanceId residue period.")
    args = parser.parse_args()

    raw_payload = json.loads(args.input.read_text())
    parameters = dict(raw_payload.get("parameters", {}))
    parameters["period"] = args.period
    payload = training_input_from_json(
        {
            **raw_payload,
            "artifactPath": str(args.artifact) if args.artifact else raw_payload.get("artifactPath"),
            "parameters": parameters,
        }
    )
    model = (
        ResidueFrequencyLootModel()
        if args.model == "residue-frequency"
        else DeterministicLootModel()
    )
    result = model.train(payload)

    payload = {
        "model": {
            "id": result.model.id,
            "version": result.model.version,
            "description": result.model.description,
        },
        "trainedAt": result.trained_at,
        "exampleCount": result.example_count,
        "validationExampleCount": result.validation_example_count,
        "artifactPath": result.artifact_path,
        "metrics": result.metrics,
    }

    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        if args.model == "deterministic":
            args.artifact.write_text(json.dumps(payload["model"], indent=2))

    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
