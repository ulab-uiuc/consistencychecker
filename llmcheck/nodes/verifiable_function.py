import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VerifiableFunction:
    code: str
    programming_language: str
    inputs: List[dict[str, Any]]  # kwargs for the function
    description: str  # New property to store the function description
    time_limit: float  # Time limit for function execution in seconds
    exec_results: List[Any] = field(default_factory=list)  # Store the results of the function execution

    def __init__(self, code: str, programming_language: str, inputs: List[dict[str, Any]], description: str, time_limit: float = 2.0) -> None:
        pattern = r'```(\w+)?'
        cleaned_code = re.sub(pattern, '', code).replace('```', '').strip()
        self.code = cleaned_code
        self.programming_language = programming_language
        self.inputs = inputs
        self.description = description
        self.exec_results = []
        self.time_limit = time_limit

    def __str__(self) -> str:
        """Pretty print the function details"""
        return (
            f"Programming Language: {self.programming_language}\n"
            f"Inputs: {self.inputs}\n"
            f"Description: {self.description}\n"
            f"Code:\n{self.code}"
        )

    def _run_with_timeout(self, func: Any, kwargs: Dict[str, Any]) -> Any:
        """Helper method to run a function with timeout"""
        result: Dict[str, Any] = {"value": None, "exception": None}

        def target() -> None:
            try:
                result["value"] = func(**kwargs)
            except Exception as e:
                result["exception"] = e

        thread = threading.Thread(target=target)
        thread.daemon = True

        thread.start()
        thread.join(timeout=self.time_limit)

        if thread.is_alive():
            # If thread is still alive, we hit the timeout
            raise TimeoutError("Function execution exceeded the time limit")

        if result["exception"]:
            raise result["exception"]

        return result["value"]

    def exec(self, catch: bool = False) -> List[Any]:
        """Execute the function and return outputs"""
        try:
            if self.programming_language not in ["python", "python3", "py", "py3", "Python", "Python3"]:
                if not catch:
                    raise ValueError(f"Unsupported language: {self.programming_language}")
                else:
                    results: List[Any] = []
                    self.exec_results = results
                    return results

            # Prepare the code for execution
            exec_globals: dict[str, Any] = {
                "input": lambda *args, **kwargs: "",
                "print": lambda *args, **kwargs: None
            }
            exec_globals_copy = exec_globals.copy()  # Copy to avoid contamination

            exec(
                'input = lambda *args, **kwargs: ""\n' +
                'print = lambda *args, **kwargs: None\n' +
                self.code, exec_globals_copy, exec_globals_copy
            )

            main_func = exec_globals_copy.get("main")
            if not main_func:
                if not catch:
                    raise ValueError("No 'main' function found in the provided code.")
                else:
                    self.exec_results = [None] * len(self.inputs)
                    return self.exec_results

            results = []
            for input_dict in self.inputs:
                try:
                    # Disable input and print
                    exec_globals_copy['input'] = lambda *args, **kwargs: ""
                    exec_globals_copy['print'] = lambda *args, **kwargs: None

                    result = self._run_with_timeout(main_func, input_dict)
                    results.append(result)
                except Exception as e:
                    if catch:
                        results.append(None)
                    else:
                        raise RuntimeError(f"{str(e)}")

            self.exec_results = results
            return results
        except Exception as e:
            if catch:
                self.exec_results = [None] * len(self.inputs)
                return self.exec_results
            else:
                raise RuntimeError(f"Execution error: {str(e)}")
