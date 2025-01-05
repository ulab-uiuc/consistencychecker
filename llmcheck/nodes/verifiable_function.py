from dataclasses import dataclass
from typing import Any, List


@dataclass
class VerifiableFunction:
    code: str
    programming_language: str
    inputs: List[dict[str, Any]] # kwargs for the function
    description: str  # New property to store the function description
    exec_results: List[List[Any]] = []  # Store the results of the function execution

    def __str__(self) -> str:
        """Pretty print the function details"""
        return (
            f"Programming Language: {self.programming_language}\n"
            f"Inputs: {self.inputs}\n"
            f"Description: {self.description}\n"  # Display the description
            f"Code:\n{self.code}"
        )

    def exec(self) -> List[Any]:  # Return the results as they are without conversion to str
        """Execute the function and return outputs"""
        if self.exec_results:
            return self.exec_results
        try:
            if self.programming_language != "python3":
                raise ValueError(f"Unsupported language: {self.programming_language}")

            # Prepare the code for execution
            exec_globals: dict[str, Any] = {}
            exec(self.code, exec_globals)

            # Extract the parameter names and their values from the input dictionaries
            main_func = exec_globals.get("main")
            if not main_func:
                raise ValueError("No 'main' function found in the provided code.")

            # If we have multiple inputs (a list of dicts), we'll call the function with each one
            results = []
            for input_dict in self.inputs:
                # Call the function with arguments unpacked from the dictionary
                result = main_func(**input_dict)
                results.append(result)  # Append result without converting to str

            return results
        except Exception as e:
            raise RuntimeError(f"Execution error: {str(e)}")
