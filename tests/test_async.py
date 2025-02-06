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

def test_simple_async_function(default_timeout: float) -> None:
    """Test a simple async function that returns a value"""
    code = """
async def main(x: int) -> None:
    await asyncio.sleep(0.1)
    return x * 2
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Simple async function test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [10, 20])

def test_async_with_complex_computation(default_timeout: float) -> None:
    """Test async function with more complex computation"""
    code = """
async def main(numbers: list) -> None:
    await asyncio.sleep(0.1)
    return sum(x * 2 for x in numbers)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"numbers": [1, 2, 3]},
            {"numbers": [4, 5, 6]}
        ],
        description="Complex async computation test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [12, 30])

def test_async_with_multiple_awaits(default_timeout: float) -> None:
    """Test async function with multiple await statements"""
    code = """
async def helper(x: int) -> None:
    await asyncio.sleep(0.1)
    return x * 2

async def main(x: int) -> None:
    result = await helper(x)
    await asyncio.sleep(0.1)
    return result + 1
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Multiple awaits test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [11, 21])

def test_async_timeout(default_timeout: float) -> None:
    """Test that async functions timeout properly"""
    code = """
async def main(x: int) -> None:
    await asyncio.sleep(3)  # Sleep longer than timeout
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

def test_async_with_exception(default_timeout: float) -> None:
    """Test async function that raises an exception"""
    code = """
async def main(x: int) -> None:
    await asyncio.sleep(0.1)
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

def test_async_with_invalid_syntax(default_timeout: float) -> None:
    """Test async function with invalid syntax"""
    code = """
async def main(x: int) -> None:
    await asyncio.sleep(0.1)
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

def test_nested_async_functions(default_timeout: float) -> None:
    """Test nested async functions"""
    code = """
async def inner(x: int) -> None:
    await asyncio.sleep(0.1)
    return x * 2

async def outer(x: int) -> None:
    await asyncio.sleep(0.1)
    result = await inner(x)
    return result + 1

async def main(x: int) -> None:
    return await outer(x)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[{"x": 5}, {"x": 10}],
        description="Nested async functions test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [11, 21])

def test_async_with_list_comprehension(default_timeout: float) -> None:
    """Test async function with list comprehension"""
    code = """
async def process_item(x: int) -> None:
    await asyncio.sleep(0.1)
    return x * 2

async def main(numbers: list) -> None:
    tasks = [process_item(x) for x in numbers]
    results = await asyncio.gather(*tasks)
    return sum(results)
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"numbers": [1, 2, 3]},
            {"numbers": [4, 5, 6]}
        ],
        description="Async list comprehension test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [12, 30])

def test_async_with_concurrent_tasks(default_timeout: float) -> None:
    """Test async function with concurrent tasks"""
    code = """
async def slow_operation(x: int) -> None:
    await asyncio.sleep(0.1)
    return x * 2

async def main(x: int, y: int) -> None:
    task1 = asyncio.create_task(slow_operation(x))
    task2 = asyncio.create_task(slow_operation(y))
    result1 = await task1
    result2 = await task2
    return result1 + result2
"""
    func = VerifiableFunction(
        code=code,
        programming_language="python",
        inputs=[
            {"x": 5, "y": 10},
            {"x": 2, "y": 3}
        ],
        description="Concurrent tasks test",
        time_limit=default_timeout
    )
    results = func.exec(catch=True)
    assert_results(results, [30, 10])
