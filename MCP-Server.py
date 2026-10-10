import os

from mcp.server.fastmcp import FastMCP


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("Directory ", path, " already exists.")

PORT = 8000

def print_stream_info(read, write, _sid, verbose=False):
    """Print stream info and SESSION_ID """
    if verbose:
        print("READ (receives FROM server):")
        print(read)
        print()

        print("WRITE (sends TO server):")
        print(write)
        print()

        print("SESSION ID:")
        print(_sid())

# Initialize FastMCP server
mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of `a` and `b`.

    Example:
        >>> add(3, 5)
        8
    """
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """
    Subtract one integer from another.

    Args:
        a (int): The number to subtract from.
        b (int): The number to subtract.

    Returns:
        int: The result of `a - b`.

    Example:
        >>> subtract(10, 4)
        6
    """
    return a - b

# @mcp.resource("file:///endpoint/{name}")
# def return_template_document(name: str) -> str:
#     """Read a document by name"""
#     return f"Document contents of {name}"

if __name__ == "__main__":
    # Run the server over stdio so the client can connect to it
    mcp.run(transport="stdio")
