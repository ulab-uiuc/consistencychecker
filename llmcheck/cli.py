import argparse
from typing import List
from datetime import datetime

import yaml

from llmcheck.core.evaluator import LLMCheck
from llmcheck.core.generator import BenchmarkGenerator


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run LLMCheck evaluations with YAML configurations.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--result_output_folder", type=str, required=False,
        help="Path to the output folder for the evaluation results. Please make sure the folder is empty or does not exist."
    )
    parser.add_argument(
        "--benchmark", type=str, required=False,
        help="Path to the YAML benchmark file. If not provided, will generate a new benchmark and test using LLMCheck."
    )
    parser.add_argument(
        "--benchmark_output", type=str, required=False,
        default=f"benchmark-results-{datetime.now().astimezone().strftime('%Y-%m-%d-%H:%M:%S')}.yaml",
        help="Path to the output file for the benchmark results."
    )
    # arg for generating benchmark only
    parser.add_argument(
        "--benchmark_only", action="store_true",
        help="If provided, will only generate benchmark and save to the benchmark_output file."
    )

    args = parser.parse_args()
    # Test invalid argument combos
    modes: List[str] = [
        "generate_benchmark_only",
        "evaluate_only",
        "generate_benchmark_and_evaluate"
    ]
    mode: str = ""
    if args.benchmark_only:
        mode = "generate_benchmark_only"
    elif args.benchmark:
        mode = "evaluate_only"
    else:
        mode = "generate_benchmark_and_evaluate"
    assert mode in modes

    parameter_combination_hint: str = """Invalid parameter combination. Please use one of the following combinations:
------------
There are a total of 3 parameter combinations:
1. Generate benchmark only.
    --config <path_to_config> 
    --benchmark_output <path_for_saving_benchmark> 
    --benchmark_only
2. Evaluate using existing benchmark.
    --config <path_to_config> 
    --benchmark <path_to_benchmark> 
    --result_output_folder <path_to_result>
3. Generate benchmark and evaluate.
    --config <path_to_config>
    --benchmark_output <path_for_saving_benchmark>
    """
    
    if mode == "generate_benchmark_only":
        assert args.config, parameter_combination_hint + "\n" + "config is required."
        assert not args.benchmark, parameter_combination_hint + "\n" + "benchmark should not be provided."
        assert not args.result_output_folder, parameter_combination_hint + "\n" + "result_output_folder should not be provided."
        assert args.benchmark_output, parameter_combination_hint + "\n" + "benchmark_output is required."
        assert args.benchmark_only, parameter_combination_hint + "\n" + "benchmark_only is required."
    elif mode == "evaluate_only":
        assert args.config, parameter_combination_hint
        assert args.benchmark, parameter_combination_hint
        assert args.result_output_folder, parameter_combination_hint
        assert not args.benchmark_output, parameter_combination_hint
        assert not args.benchmark_only, parameter_combination_hint
    elif mode == "generate_benchmark_and_evaluate":
        assert args.config, parameter_combination_hint
        assert not args.benchmark, parameter_combination_hint
        assert args.result_output_folder, parameter_combination_hint
        assert args.benchmark_output, parameter_combination_hint
        assert not args.benchmark_only, parameter_combination_hint

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    step_generate_benchmark: bool = False if mode == "evaluate_only" else True
    step_evaluate: bool = False if mode == "generate_benchmark_only" else True

    forest: List[dict] = []

    if step_generate_benchmark:
        print("[INFO] Generating benchmark...")

        # print("evaluator model name: ", config.get("evaluator").get("model_name"))
        # print("evaluator model api base: ", config.get("evaluator").get("api_base"))
        # print("evaluator model temperature: ", config.get("evaluator").get("temperature"))
        # print("llm max new tokens: ", config.get("llm_max_new_tokens"))
        
        # print("forest size: ", config.get("forest_size", 1))
        # print("root_node_constraints: ", config.get("root_node_constraints"))
        # print("operation_generation_prompt: ", config.get("operation_generation_prompt"))
        # print("n_operations: ", config.get("n_operations"))

        # print("benchmark_output: ", args.benchmark_output)

        benchmark_generator = BenchmarkGenerator(
            evaluator_model=config.get("evaluator").get("model_name"),
            evaluator_model_api_base=config.get("evaluator").get("api_base"),
            evaluator_model_temperature=config.get("evaluator").get("temperature"),
            llm_max_new_tokens=config.get("llm_max_new_tokens"),
        )

        # get forest size
        forest_size = config.get("forest_size")

        for tree_idx in range(forest_size):
            print(f"[INFO] Generating tree {tree_idx + 1} / {forest_size}", end="\r")
            # Generate benchmark
            retry_times: int = 0
            while True:
                try:
                    root, operations = benchmark_generator.generate_benchmark(
                        constraints=config.get("root_node_constraints"),
                        prompt=config.get("operation_generation_prompt"),
                        n_operations=config.get("n_operations")
                    )
                    break
                except Exception as e:
                    print(e)
                    retry_times += 1
                    print(f"[INFO] Retry: {retry_times}")
                    continue
            forest.append({"operations": operations, "root": root})
        
        print(f"[INFO] Generating tree {forest_size} / {forest_size}")
        
        # write to benchmark file
        with open(args.benchmark_output, "w") as file:
            yaml.dump({"forest": forest}, file, default_flow_style=None, sort_keys=False)
        print(f"[INFO] Benchmark saved to {args.benchmark_output}")
    else:
        # load from benchmark file
        # with open(args.benchmark, "r") as file:
        #     benchmark = yaml.safe_load(file)
        #     forest = benchmark.get("forest")
        #     # check format if each element has root and operations keys
        #     for tree in forest:
        #         assert "root" in tree, "Invalid benchmark format."
        #         assert "operations" in tree, "Invalid benchmark format."
        pass

    if not step_evaluate:
        return

    return

    # Extract parameters
    similarity_config = config.get("similarity_config")
    num_of_samples = config.get("num_of_samples")

    for tree_idx, tree in enumerate(forest):
        root = tree["root"]
        operations = tree["operations"]
        llmcheck_instance = LLMCheck(
            evaluatee_model=config.get("evaluatee").get("model_name"),
            evaluatee_api_base=config.get("evaluatee").get("api_base"),
            evaluatee_model_temperature=config.get("evaluatee").get("temperature"),
            similarity_config=similarity_config,
            max_depth=config.get("max_depth"),
            operation_code_format_enforce_prompt=config.get("operation_code_format_enforce_prompt"),
            llm_max_new_tokens=config.get("llm_max_new_tokens"),
            retry_max=config.get("retry_max"),
            time_limit=config.get("time_limit"),
        )
        for sample_idx in range(num_of_samples):
            print(f"Sample {sample_idx + 1} / {num_of_samples}")
            results = llmcheck_instance.evaluate(
                root=root,
                operations=[tuple(op) for op in operations],
                distance=config.get("distance")
            )
            # Save results to YAML file
            with open(f"{args.result_output_folder}/{tree_idx}_{sample_idx}.yaml", "w") as file:
                yaml.dump(results, file, default_flow_style=None, sort_keys=False)

if __name__ == "__main__":
    cli()
