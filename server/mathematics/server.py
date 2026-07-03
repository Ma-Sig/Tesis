"""
Mathematics MCP Server

Provides mathematical operations and calculations through the Model Context Protocol.
"""

import argparse
import math
import statistics
import pandas as pd
from typing import List
from fastmcp import FastMCP

# Create FastMCP server instance
mcp = FastMCP("Mathematics Server")


@mcp.tool()
def analyze_csv(filename: str) -> dict:
    """Analyze uploaded CSV file."""
    df = pd.read_csv(f"/data/uploads/{filename}")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "stats": df.describe().to_dict(),
    }


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        Difference of a and b
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Quotient of a and b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.

    Args:
        base: Base number
        exponent: Exponent

    Returns:
        base^exponent
    """
    return math.pow(base, exponent)


@mcp.tool()
def square_root(number: float) -> float:
    """Calculate square root of a number.

    Args:
        number: Number to find square root of

    Returns:
        Square root of the number

    Raises:
        ValueError: If number is negative
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)


@mcp.tool()
def mean(numbers: List[float]) -> float:
    """Calculate arithmetic mean of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Mean of the numbers

    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return statistics.mean(numbers)


@mcp.tool()
def median(numbers: List[float]) -> float:
    """Calculate median of a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Median of the numbers

    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate median of empty list")
    return statistics.median(numbers)


@mcp.tool()
def standard_deviation(numbers: List[float]) -> float:
    """Calculate standard deviation of a list of numbers.

    Args:
        numbers: List of numbers (requires at least 2 values)

    Returns:
        Standard deviation of the numbers

    Raises:
        ValueError: If the list has fewer than 2 values
    """
    if len(numbers) < 2:
        raise ValueError("Standard deviation requires at least 2 values")
    return statistics.stdev(numbers)


@mcp.tool()
def percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole.

    Args:
        part: Part value
        whole: Whole value

    Returns:
        Percentage

    Raises:
        ValueError: If whole is zero
    """
    if whole == 0:
        raise ValueError("Cannot calculate percentage when whole is zero")
    return (part / whole) * 100


@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Greatest common divisor
    """
    return math.gcd(a, b)


@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate least common multiple of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Least common multiple
    """
    return math.lcm(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mathematics MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (stdio for local, http for network)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (for HTTP)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(f"Starting Mathematics MCP Server on http://{args.host}:{args.port}/mcp")
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
