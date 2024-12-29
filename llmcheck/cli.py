import argparse
from datetime import datetime

import yaml

from llmcheck.core.evaluator import LLMCheck


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run LLMCheck evaluations with YAML configurations.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--result", type=str, required=False,
        default=f"results-{datetime.now().astimezone().strftime('%Y-%m-%d-%H:%M:%S')}.yaml",
        help="Path to the output file for the results."
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract parameters
    similarity_config = config.get("similarity_config")
    constraints = config.get("constraints")
    root = config.get("root", "")
    operations = config.get("operations", [])
    distance = config.get("distance")

    # Initialize LLMCheck
    checker = LLMCheck(
        evaluator_model=config.get("evaluator").get("model_name"),
        evaluator_api_base=config.get("evaluator").get("api_base"),
        target_model=config.get("target").get("model_name"),
        target_api_base=config.get("target").get("api_base"),
        similarity_config=similarity_config,
        max_depth=config.get("max_depth"),
        n_operations=config.get("n_operations")
    )

    # Run evaluation
    results = checker.evaluate(
        constraints=constraints,
        root=root,
        operations=[tuple(op) for op in operations],
        distance=distance
    )

    # Save results to YAML file
    with open(args.result, "w") as file:
        yaml.dump(results, file, default_flow_style=False)


if __name__ == "__main__":
    cli()
