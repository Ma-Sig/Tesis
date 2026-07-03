# Mathematics MCP Server

Provides mathematical operations and calculations through the Model Context Protocol.

## Available Tools

### Basic Arithmetic

- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract b from a
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide a by b
- `power(base, exponent)` - Raise base to exponent
- `square_root(number)` - Calculate square root

### Advanced Operations

- `factorial(n)` - Calculate factorial
- `gcd(a, b)` - Greatest common divisor
- `lcm(a, b)` - Least common multiple
- `percentage(part, whole)` - Calculate percentage

### Statistics

- `mean(numbers)` - Arithmetic mean
- `median(numbers)` - Median value
- `standard_deviation(numbers)` - Standard deviation

## Usage Example

This server runs via stdio transport. It's automatically managed by the AMALIA backend.

## Configuration

Register in `/backend/mcp_servers_config.json`:

```json
{
  "mathematics": {
    "name": "Mathematics Server",
    "path": "/app/mcp_servers/mathematics/server.py",
    "command": "python",
    "args": [],
    "allowed_paths": ["/data/uploads"],
    "env": { "PYTHONPATH": "/app" },
    "description": "Mathematical operations and statistical calculations"
  }
}
```
