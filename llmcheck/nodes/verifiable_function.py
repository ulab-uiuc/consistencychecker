import re
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class VerifiableFunction:
    code: str
    programming_language: str
    inputs: List[dict[str, Any]] # kwargs for the function
    description: str  # New property to store the function description
    exec_results: List[Any] = field(default_factory=list)  # Store the results of the function execution
    def __init__(self, code: str, programming_language: str, inputs: List[dict[str, Any]], description: str, exec_results: List[Any] = field(default_factory=list)) -> None:
        pattern = r'```(\w+)?'
        cleaned_code = re.sub(pattern, '', code).replace('```', '').strip()
        self.code = cleaned_code
        self.programming_language = programming_language
        self.inputs = inputs
        self.description = description
        self.exec_results = []
    def __str__(self) -> str:
        """Pretty print the function details"""
        return (
            f"Programming Language: {self.programming_language}\n"
            f"Inputs: {self.inputs}\n"
            f"Description: {self.description}\n"  # Display the description
            f"Code:\n{self.code}"
        )

    def exec(self, catch:bool=False) -> List[Any]:  # Return the results as they are without conversion to str
        """Execute the function and return outputs"""
        try:
            if self.programming_language not in ["python", "python3", "py", "py3", "Python", "Python3"]: # just in case the LLM is bad at following instructions
                if not catch:
                    raise ValueError(f"Unsupported language: {self.programming_language}")
                else:
                    results: List[Any] = []
                    print(f"[ERROR] Unsupported language: {self.programming_language}")
                    return results

            # Prepare the code for execution
            exec_globals: dict[str, Any] = {}
            exec(self.code, exec_globals)

            # Extract the parameter names and their values from the input dictionaries
            main_func = exec_globals.get("main")
            if not main_func:
                if not catch:
                    raise ValueError("No 'main' function found in the provided code.")
                else:
                    results = []
                    print("[ERROR] No 'main' function found in the provided code.")
                    return results
            # If we have multiple inputs (a list of dicts), we'll call the function with each one
            results = []
            for input_dict in self.inputs:
                # Call the function with arguments unpacked from the dictionary
                try:
                    result = main_func(**input_dict)
                    results.append(result)  # Append result without converting to str
                except Exception as e:
                    results.append(f"Error: {str(e)}")
                    if catch:
                        print(f"Error: {str(e)}")
                    else:
                        raise RuntimeError(f"Execution error: {str(e)}")

            return results
        except Exception as e:
            if catch:
                print(f"Error: {str(e)}")
                return []
            else:
                raise RuntimeError(f"Execution error: {str(e)}")
