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
        return (
            f"Programming Language: {self.programming_language}\n"
            f"Inputs: {self.inputs}\n"
            f"Description: {self.description}\n"
            f"Code:\n{self.code}"
        )

    async def _run_async_function(self, main_func: Any, input_dict: dict[str, Any]) -> Any:
        """Helper function to execute async functions."""
        try:
            return await main_func(**input_dict)
        except Exception:
            return None

    def _run_function(self, input_dict: dict[str, Any]) -> Any:
        """Helper function to execute the code in a separate process."""
        def worker(code: str, input_dict: Dict[str, Any], queue: Any) -> None:
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

                queue.put(result)
                loop.close()
            except Exception:
                queue.put(None)
            finally:
                if 'loop' in locals() and not loop.is_closed():
                    loop.close()

        queue: Any = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker, args=(self.code, input_dict, queue))
        process.start()
        process.join(timeout=self.time_limit)

        if process.is_alive():
            process.terminate()
            process.join()
            return None
        elif queue.empty():
            return None
        else:
            return queue.get()

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
                result = self._run_function(input_dict)
                if isinstance(result, Exception):
                    if catch:
                        results.append(None)
                    else:
                        raise result
                else:
                    results.append(result)

            self.exec_results = results
            return results
        except Exception as e:
            if catch:
                self.exec_results = [None] * len(self.inputs)
                return self.exec_results
            else:
                raise RuntimeError(f"Execution error: {str(e)}")
