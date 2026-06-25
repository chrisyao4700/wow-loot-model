from __future__ import annotations

import argparse
import json
from pathlib import Path

from sea_of_blossoms_model.catalog import load_catalog
from sea_of_blossoms_model.contracts import LootTrainingInput
from sea_of_blossoms_model.serialization import training_input_to_json
from sea_of_blossoms_model.training_data import generate_deterministic_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare observed loot training data.")
    parser.add_argument("--catalog", type=Path, required=True, help="Raid catalog JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Training data JSON output.")
    parser.add_argument("--raid-id", type=int, action="append", help="Numeric raid id to include.")
    parser.add_argument("--instance-start", type=int, default=10000000)
    parser.add_argument("--instance-count", type=int, default=100)
    parser.add_argument("--instance-step", type=int, default=1)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    selected_raid_ids = set(args.raid_id or [raid.numeric_id for raid in catalog])
    raids = tuple(raid for raid in catalog if raid.numeric_id in selected_raid_ids)

    if not raids:
        raise SystemExit("No raids matched the requested --raid-id values.")

    examples = generate_deterministic_observations(
        raids,
        instance_start=args.instance_start,
        instance_count=args.instance_count,
        instance_step=args.instance_step,
    )
    payload = LootTrainingInput(
        examples=examples,
        parameters={
            "source": "deterministic-baseline",
            "instanceStart": args.instance_start,
            "instanceCount": args.instance_count,
            "instanceStep": args.instance_step,
            "raidIds": ",".join(str(raid.numeric_id) for raid in raids),
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(training_input_to_json(payload), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
