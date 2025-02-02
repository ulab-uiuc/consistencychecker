import argparse
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List

import colorama
import yaml

from llmcheck.core.evaluator import LLMCheck
from llmcheck.core.generator import BenchmarkGenerator

INFO_PLAIN: str = "INFO"
INFO_GREEN: str = colorama.Fore.GREEN + "INFO" + colorama.Style.RESET_ALL


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

    parameter_combination_hint: str = f"""Invalid parameter combination. Please use one of the following combinations:
------------
There are a total of 3 parameter combinations:
1. Generate benchmark only.
    {colorama.Fore.BLUE}--config <path_to_config>{colorama.Style.RESET_ALL}
    {colorama.Fore.BLUE}--benchmark_output <path_for_saving_benchmark>{colorama.Style.RESET_ALL}
    {colorama.Fore.BLUE}--benchmark_only{colorama.Style.RESET_ALL}
2. Evaluate using existing benchmark.
    {colorama.Fore.BLUE}--config <path_to_config>{colorama.Style.RESET_ALL}
    {colorama.Fore.BLUE}--benchmark <path_to_benchmark>{colorama.Style.RESET_ALL}
    {colorama.Fore.BLUE}--result_output_folder <path_for_saving_results>{colorama.Style.RESET_ALL}
3. Generate benchmark and evaluate.
    {colorama.Fore.BLUE}--config <path_to_config>{colorama.Style.RESET_ALL}
    {colorama.Fore.BLUE}--result_output_folder <path_for_saving_results>{colorama.Style.RESET_ALL}
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

    forest: List[Dict[str, Any]] = []

    # get forest size
    forest_size = config.get("forest_size")

    if step_generate_benchmark:
        # if benchmark file exists, raise an error
        if os.path.exists(args.benchmark_output):
            raise Exception("The benchmark file already exists. To avoid overwriting, please provide a different file name.")

        print(f"[{INFO_PLAIN}] Generating benchmark...")

        benchmark_generator = BenchmarkGenerator(
            evaluator_model=config.get("evaluator").get("model_name"),
            evaluator_model_api_base=config.get("evaluator").get("api_base"),
            evaluator_model_temperature=config.get("evaluator").get("temperature"),
            llm_max_new_tokens=config.get("llm_max_new_tokens"),
            time_limit=config.get("time_limit")
        )

        retry_max: int = config.get("retry_max")

        for tree_idx in range(forest_size):
            print(f"[{INFO_PLAIN}] Generating tree {tree_idx + 1} / {forest_size}")
            # Generate benchmark
            retry_times: int = 0
            while retry_times < retry_max:
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
                    print(f"[{INFO_PLAIN}] Generating tree {tree_idx + 1} / {forest_size} failed. Retrying... ({retry_times} / {retry_max})")
                    continue
            if retry_times == retry_max:
                print(f"[ERROR] Failed to generate tree {tree_idx + 1} / {forest_size}")
                raise Exception(f"Benchmark generation failed.\nCrank up the retry_max(now {retry_max}) would help.")
            forest.append({"operations": operations, "root": root})

        # write to benchmark file
        with open(args.benchmark_output, "w") as file:
            yaml.dump({"forest": forest}, file, default_flow_style=None, sort_keys=False)
        print(f"[{INFO_GREEN}] Benchmark saved to {args.benchmark_output}")
    else:
        # load from benchmark file
        with open(args.benchmark, "r") as file:
            benchmark = yaml.safe_load(file)
            forest = benchmark.get("forest")
            # check format if each element has root and operations keys
            for tree in forest:
                assert "root" in tree, "Invalid benchmark format."
                assert "operations" in tree, "Invalid benchmark format."
        pass

    if not step_evaluate:
        return

    timestamp_run_start: datetime = datetime.now()
    run_passes: int = 0

    # if the target folder does not exist, create it
    if args.result_output_folder:
        if not os.path.exists(args.result_output_folder):
            os.makedirs(args.result_output_folder)
    # if files in the target folder exist, raise an error
    if args.result_output_folder:
        if os.path.exists(args.result_output_folder) and os.listdir(args.result_output_folder):
            raise Exception("The target folder is not empty. To avoid overwriting, please provide an empty folder.")


    # Extract parameters
    similarity_config = config.get("similarity_config")
    num_of_samples = config.get("num_of_samples")

    full_avg_metrics_collect = defaultdict(list)
    for tree_idx, tree in enumerate(forest):
        root_original = tree["root"].copy()
        root_original.pop("exec_results")
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
        root = deepcopy(root_original)
        for sample_idx in range(num_of_samples):
            time_eta_str: str = "N/A"
            if run_passes > 0:
                time_eta: timedelta = (datetime.now() - timestamp_run_start) / run_passes * (forest_size * num_of_samples - run_passes)
                time_eta_str = str(timedelta(seconds=int(time_eta.total_seconds())))
                print(f"Estimated time remaining: {time_eta_str}")
            print(f"{colorama.Fore.BLUE}Evaluating tree {tree_idx + 1} / {forest_size}, {colorama.Fore.GREEN}Sample {sample_idx + 1} / {num_of_samples}, {colorama.Fore.RED}ETA {time_eta_str}{colorama.Style.RESET_ALL}")
            results = llmcheck_instance.evaluate(
                root=root,
                operations=operations,
                distance=config.get("l_scores")
            )
            for key in results["metrics"]:
                if "AVG" in key:
                    full_avg_metrics_collect[key].append(results["metrics"][key])
            # Save results to YAML file
            with open(f"{args.result_output_folder}/tree_{tree_idx}_sample_{sample_idx}.yaml", "w") as file:
                yaml.dump(results, file, default_flow_style=None, sort_keys=False)
            run_passes += 1
    # save all values, avg, and std, of the full_avg_metrics_collect
    full_avg_metrics = {}
    for key in full_avg_metrics_collect:
        assert len(full_avg_metrics_collect[key]) == forest_size * num_of_samples
    for key in full_avg_metrics_collect:
        full_avg_metrics[key] = {
            "values": full_avg_metrics_collect[key],
            "avg": sum(full_avg_metrics_collect[key]) / len(full_avg_metrics_collect[key]),
            "std": sum((x - (sum(full_avg_metrics_collect[key]) / len(full_avg_metrics_collect[key]))) ** 2 for x in full_avg_metrics_collect[key]) / len(full_avg_metrics_collect[key])
        }
    with open(f"{args.result_output_folder}/full_avg_metrics.yaml", "w") as file:
        yaml.dump(full_avg_metrics, file, default_flow_style=None, sort_keys=False)
    print(f"[{INFO_GREEN}] Evaluation results saved to {args.result_output_folder}")
    return

if __name__ == "__main__":
    # colorama init
    colorama.init(autoreset=True)
    # for yaml dumping of arbitrarily large integers
    sys.set_int_max_str_digits(2147483647)
    # def str_presenter(dumper: yaml.Dumper, data: str) -> yaml.nodes.ScalarNode:
    #     # Check if string contains newlines
    #     if '\n' in data:
    #         return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    #     return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    # yaml.add_representer(str, str_presenter)
    # TOKENIZERS_PARALLELISM = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cli()
