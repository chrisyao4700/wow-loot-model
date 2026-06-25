from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sea_of_blossoms_model.baselines import DeterministicLootModel
from sea_of_blossoms_model.models import ResidueFrequencyLootModel
from sea_of_blossoms_model.serialization import (
    prediction_input_from_api,
    prediction_output_to_api,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline loot model prediction.")
    parser.add_argument("--input", type=Path, help="Prediction input JSON. Defaults to stdin.")
    parser.add_argument("--output", type=Path, help="Prediction output JSON. Defaults to stdout.")
    parser.add_argument("--artifact", type=Path, help="Trained model artifact JSON.")
    args = parser.parse_args()

    raw_payload = args.input.read_text() if args.input else sys.stdin.read()
    payload = prediction_input_from_api(json.loads(raw_payload))
    model = (
        ResidueFrequencyLootModel.from_artifact_path(args.artifact)
        if args.artifact
        else DeterministicLootModel()
    )
    output = prediction_output_to_api(model.predict(payload))
    encoded = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        args.output.write_text(f"{encoded}\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
