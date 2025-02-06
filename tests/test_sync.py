from typing import Any, List

import pytest

from llmcheck.nodes.verifiable_function import VerifiableFunction

# Import your VerifiableFunction class here
# from verifiable_function import VerifiableFunction

@pytest.fixture
def default_timeout() -> float:
    return 2.0

def assert_results(results: List[Any], expected: List[Any]) -> None:
    """Helper to assert results match expected values"""
    assert len(results) == len(expected)
    for result, expect in zip(results, expected):
        assert result == expect

def test_simple_sync_function(default_timeout: float) -> None:
    """Test a simple sync function that returns a value"""
    code = """
import time
def main(x: int) -> None:
    time.sleep(0.1)
    return x * 2
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Simple sync function test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [10, 20])

def test_sync_with_complex_computation(default_timeout: float) -> None:
    """Test sync function with more complex computation"""
    code = """
import time
def main(numbers: list) -> None:
    time.sleep(0.1)
    return sum(x * 2 for x in numbers)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"numbers": [1, 2, 3]},
            {"numbers": [4, 5, 6]}
        ],
        description="Complex sync computation test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [12, 30])

def test_sync_with_multiple_functions(default_timeout: float) -> None:
    """Test sync function with multiple function calls"""
    code = """
import time
def helper(x: int) -> None:
    time.sleep(0.1)
    return x * 2

def main(x: int) -> None:
    result = helper(x)
    time.sleep(0.1)
    return result + 1
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Multiple functions test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [11, 21])

def test_sync_timeout(default_timeout: float) -> None:
    """Test that sync functions timeout properly"""
    code = """
import time
def main(x: int) -> None:
    time.sleep(3)  # Sleep longer than timeout
    return x
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}],
        description="Timeout test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert results[0] is None

def test_sync_with_exception(default_timeout: float) -> None:
    """Test sync function that raises an exception"""
    code = """
import time
def main(x: int) -> None:
    time.sleep(0.1)
    raise ValueError("Test error")
    return x
"""
    # Test with catch=True
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}],
        description="Exception test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert results[0] is None

    # Test with catch=False
    with pytest.raises(ValueError):
        func.exec(catch=False)

def test_sync_with_invalid_syntax(default_timeout: float) -> None:
    """Test sync function with invalid syntax"""
    code = """
import time
def main(x: int) -> None:
    time.sleep(0.1)
    return x * 2
    invalid syntax here
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}],
        description="Invalid syntax test",
        time_limit=default_timeout
    )
    # Test with catch=True
    results = func.exec(catch=True)
    assert results[0] is None

    # Test with catch=False
    with pytest.raises(SyntaxError):
        func.exec(catch=False)

def test_nested_sync_functions(default_timeout: float) -> None:
    """Test nested sync functions"""
    code = """
import time
def inner(x: int) -> None:
    time.sleep(0.1)
    return x * 2

def outer(x: int) -> None:
    time.sleep(0.1)
    result = inner(x)
    return result + 1

def main(x: int) -> None:
    return outer(x)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Nested sync functions test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [11, 21])

def test_sync_with_list_comprehension(default_timeout: float) -> None:
    """Test sync function with list comprehension"""
    code = """
import time
def process_item(x: int) -> None:
    time.sleep(0.1)
    return x * 2

def main(numbers: list) -> None:
    results = [process_item(x) for x in numbers]
    return sum(results)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"numbers": [1, 2, 3]},
            {"numbers": [4, 5, 6]}
        ],
        description="List comprehension test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [12, 30])

def test_sync_with_parallel_operations(default_timeout: float) -> None:
    """Test sync function with multiple operations"""
    code = """
import time
def slow_operation(x: int) -> None:
    time.sleep(0.1)
    return x * 2

def main(x: int, y: int) -> None:
    result1 = slow_operation(x)
    result2 = slow_operation(y)
    return result1 + result2
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"x": 5, "y": 10},
            {"x": 2, "y": 3}
        ],
        description="Multiple operations test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [30, 10])

def test_sync_with_recursive_function(default_timeout: float) -> None:
    """Test sync function with recursion"""
    code = """
def factorial(n: int) -> None:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def main(n: int) -> None:
    return factorial(n)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"n": 5}, {"n": 3}],
        description="Recursive function test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [120, 6])

def test_sync_with_generator(default_timeout: float) -> None:
    """Test sync function with generator"""
    code = """
def generate_squares(n: int):
    for i in range(n):
        yield i * i

def main(n: int) -> None:
    return list(generate_squares(n))
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"n": 4}, {"n": 3}],
        description="Generator function test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [[0, 1, 4, 9], [0, 1, 4]])
