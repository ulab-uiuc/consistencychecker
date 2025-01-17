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
    root = config.get("root", {})
    operations = config.get("operations", [])
    distance = config.get("distance")
    prompt_template = config.get("prompt_template")

    # Initialize LLMCheck
    checker = LLMCheck(
        evaluator_model=config.get("evaluator").get("model_name"),
        evaluator_api_base=config.get("evaluator").get("api_base"),
        evaluator_model_temperature=config.get("evaluator").get("temperature"),
        evaluatee_model=config.get("evaluatee").get("model_name"),
        evaluatee_api_base=config.get("evaluatee").get("api_base"),
        evaluatee_model_temperature=config.get("evaluatee").get("temperature"),
        similarity_config=similarity_config,
        max_depth=config.get("max_depth"),
        n_operations=config.get("n_operations"),
        operation_code_format_enforce_prompt=config.get("operation_code_format_enforce_prompt"),
        llm_max_new_tokens=config.get("llm_max_new_tokens"),
        retry_max=config.get("retry_max"),
    )

    # Run evaluation
    results = checker.evaluate(
        constraints=constraints,
        prompt_template=prompt_template,
        root=root,
        operations=[tuple(op) for op in operations],
        distance=distance
    )

    print("Results:")
    for key in results["metrics"]:
        if "AVG" in key:
            print(f"    {key}: {results['metrics'][key]}")

    # Save results to YAML file
    with open(args.result, "w") as file:
        yaml.dump(results, file, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    cli()
