import json
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

TOOL_NAME = "customer_lookup"

DESCRIPTION = (
    "Looks up a customer by email (e.g. jane@example.com) or customer ID (e.g. CUST-10042) "
    "in the customer database. "
    "The failure_mode parameter simulates specific outcomes for testing error handling; "
    "leave it as 'success' for a normal lookup."
)

ErrorCategory = Literal["transient", "validation", "business", "permission"]


def build_error(category: ErrorCategory, retryable: bool, description: str) -> CallToolResult:
    """Every tool error goes through here, so all of them carry the same three fields."""
    error = {
        "errorCategory": category,
        "isRetryable": retryable,
        "description": description,
    }
    # The model reads `content`, so the metadata goes there as JSON.
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(error))],
        is_error=True,
    )


def customer_lookup(
    identifier: str,
    failure_mode: Literal["success", "not_found", ErrorCategory] = "success",
) -> CallToolResult:
    if failure_mode == "transient":
        return build_error(
            "transient",
            True,
            "The customer database timed out after 5 seconds; the query did not execute. "
            "The request is valid and should succeed on retry.",
        )

    if failure_mode == "validation":
        return build_error(
            "validation",
            False,
            f"Invalid identifier format: '{identifier}'. Expected an email (user@domain.com) "
            "or a customer ID (CUST-NNNNN). Correct the format and call again.",
        )

    if failure_mode == "business":
        return build_error(
            "business",
            False,
            "Refund of £750 exceeds the £500 automatic approval limit. "
            "Do not retry; escalate to a manager with the refund details.",
        )

    if failure_mode == "permission":
        return build_error(
            "permission",
            False,
            "The current service account lacks access to financial records. "
            "Do not retry with the same credentials; escalate to a senior agent "
            "or request elevated access.",
        )

    if failure_mode == "not_found":
        return CallToolResult(
            content=[TextContent(type="text", 
                text=json.dumps({
                    "text": f"No customer found matching identifier {identifier}. The query executed successfully but returned no matches.",
                    "resultCount": 0
                })
            )],
            is_error=False,
        )

    return CallToolResult(
        content=[TextContent(type="text", text=f"Mode: {failure_mode}. Customer data for {identifier}")],
    )


mcp = MCPServer("customer-db", log_level="ERROR")
mcp.tool(name=TOOL_NAME, description=DESCRIPTION)(customer_lookup)



if __name__ == "__main__":
    mcp.run(transport="stdio")
