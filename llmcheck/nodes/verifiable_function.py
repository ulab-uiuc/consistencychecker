import multiprocessing
import re
from dataclasses import dataclass, field
from typing import Any, List


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
            f"Description: {self.description}\n"  # Display the description
            f"Code:\n{self.code}"
        )

    def _run_function(self, input_dict: dict[str, Any], queue: Any) -> None:
        """Helper function to execute the code in a separate process."""
        try:
            # Create a new clean dictionary to encapsulate the function execution
            exec_globals: dict[str, Any] = {
                "input": lambda *args, **kwargs: "",
                "print": lambda *args, **kwargs: None
            }

            # Execute the code in an isolated context
            exec(
                'input = lambda *args, **kwargs: ""\n' +
                'print = lambda *args, **kwargs: None\n' +
                self.code, exec_globals, exec_globals
            )

            # Extract the main function
            main_func = exec_globals.get("main")
            if not main_func:
                raise ValueError("No 'main' function found in the provided code.")

            # Call the function with the input dictionary
            result = main_func(**input_dict)
            queue.put(result)  # Put the result in the queue
        except Exception as e:
            queue.put(e)  # Put the exception in the queue if something goes wrong

    def exec(self, catch: bool = False) -> List[Any]:
        """Execute the function and return outputs."""
        try:
            if self.programming_language not in ["python", "python3", "py", "py3", "Python", "Python3"]:
                if not catch:
                    raise ValueError(f"Unsupported language: {self.programming_language}")
                else:
                    results: List[Any] = []
                    self.exec_results = results
                    return results

            results = []
            for input_dict in self.inputs:
                # Use a multiprocessing Queue to communicate results between processes
                queue:Any = multiprocessing.Queue()

                # Create and start the process
                process = multiprocessing.Process(
                    target=self._run_function,
                    args=(input_dict, queue)
                )
                process.start()

                # Wait for the process to complete or timeout
                process.join(timeout=self.time_limit)

                if process.is_alive():
                    # If the process is still alive, it exceeded the time limit
                    process.terminate()  # Terminate the process
                    process.join()  # Ensure the process is cleaned up
                    results.append(None)  # Append None to indicate a timeout
                else:
                    # Process completed within the time limit
                    if not queue.empty():
                        result = queue.get()
                        if isinstance(result, Exception):
                            if catch:
                                results.append(None)
                            else:
                                raise result
                        else:
                            results.append(result)
                    else:
                        results.append(None)  # No result was produced

            self.exec_results = results
            return results
        except Exception as e:
            if catch:
                self.exec_results = [None] * len(self.inputs)
                return self.exec_results
            else:
                raise RuntimeError(f"Execution error: {str(e)}")
