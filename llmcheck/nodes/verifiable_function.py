import asyncio
import inspect
import multiprocessing
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VerifiableFunction:
    code: str
    programming_language: str
    inputs: List[dict[str, Any]]
    description: str
    time_limit: float
    exec_results: List[Any] = field(default_factory=list)

    def __init__(self, code: str, programming_language: str, inputs: List[dict[str, Any]], description: str, time_limit: float = 2.0, exec_results: List[Any]=[]) -> None:
        pattern = r'```(\w+)?'
        cleaned_code = re.sub(pattern, '', code).replace('```', '').strip()
        self.code = cleaned_code
        self.programming_language = programming_language
        self.inputs = inputs
        self.description = description
        self.exec_results = []
        self.time_limit = time_limit

    def __str__(self) -> str:
        return (
            f"Programming Language: {self.programming_language}\n"
            f"Inputs: {self.inputs}\n"
            f"Description: {self.description}\n"
            f"Code:\n{self.code}"
        )

    async def _run_async_function(self, main_func: Any, input_dict: dict[str, Any]) -> Any:
        """Helper function to execute async functions."""
        result = await main_func(**input_dict)
        # Handle async generators
        if inspect.isasyncgen(result):
            result = [item async for item in result]
        return result

    def _run_function(self, input_dict: dict[str, Any]) -> Any:
        """Helper function to execute the code in a separate process."""
        def worker(code: str, input_dict: Dict[str, Any], result_queue: Any, error_queue: Any) -> None:
            try:
                # Create a new event loop for this process
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                exec_globals: dict[str, Any] = {
                    "asyncio": asyncio,
                    "input": lambda *args, **kwargs: "",
                    "print": lambda *args, **kwargs: None
                }

                exec(
                    'input = lambda *args, **kwargs: ""\n' +
                    'print = lambda *args, **kwargs: None\n' +
                    code, exec_globals, exec_globals
                )

                main_func = exec_globals.get("main")
                if not main_func:
                    raise ValueError("No 'main' function found in the provided code.")

                if inspect.iscoroutinefunction(main_func):
                    # Run async function
                    result = loop.run_until_complete(self._run_async_function(main_func, input_dict))
                else:
                    # Run synchronous function
                    result = main_func(**input_dict)
                    # Convert generator to list if necessary
                    if inspect.isgenerator(result):
                        result = list(result)
                    # Handle nested generators in lists/tuples
                    elif isinstance(result, (list, tuple)):
                        result = [list(x) if inspect.isgenerator(x) else x for x in result]

                result_queue.put(result)

            except Exception as e:
                error_queue.put((type(e), str(e)))
            finally:
                if 'loop' in locals() and not loop.is_closed():
                    loop.close()

        result_queue: Any = multiprocessing.Queue()
        error_queue: Any = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=worker,
            args=(self.code, input_dict, result_queue, error_queue)
        )

        process.start()
        process.join(timeout=self.time_limit)

        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"Function execution exceeded time limit of {self.time_limit} seconds")

        if not error_queue.empty():
            error_type, error_message = error_queue.get()
            if error_type is SyntaxError:
                raise SyntaxError(error_message)
            elif error_type is ValueError:
                raise ValueError(error_message)
            else:
                raise RuntimeError(error_message)

        if result_queue.empty():
            raise RuntimeError("Function execution failed without returning a result")

        return result_queue.get()

    def exec(self, catch: bool = False) -> List[Any]:
        """Execute the function and return outputs."""
        if self.programming_language not in ["python", "python3", "py", "py3", "Python", "Python3"]:
            if not catch:
                raise ValueError(f"Unsupported language: {self.programming_language}")
            return []

        results = []
        for input_dict in self.inputs:
            try:
                result = self._run_function(input_dict)
                results.append(result)
            except Exception:
                if catch:
                    results.append(None)
                else:
                    raise

        self.exec_results = results
        return results
