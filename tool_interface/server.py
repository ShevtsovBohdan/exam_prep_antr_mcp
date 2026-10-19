from mcp.server.mcpserver import MCPServer

CUSTOMER = "retrieve_customer_info"
ORDER = "retrieve_order"

# Step 1 baseline: deliberately vague.
VAGUE = {
    CUSTOMER: "Retrieves customer information",
    ORDER: "Retrieves order details",
}

# Step 3: purpose, accepted inputs, example queries, boundaries.
# Examples are intentionally different from the harness test queries.
PRODUCTION = {
    CUSTOMER: (
        "Looks up a customer's account profile: name, contact details, account status "
        "(active, suspended or closed) and loyalty tier. "
        "Accepts a customer email (e.g. jane@example.com), phone number (e.g. 555-0123) "
        "or customer ID (e.g. C-1042). "
        "Use it for questions about who the customer is or the state of their account, "
        "e.g. 'Who owns the account with phone 555-0199?' or 'Is this customer a Gold member?'. "
        "Do NOT use it for anything about orders, purchases, packages, shipping, delivery "
        "or refunds - use retrieve_order for those, even when the user only gives an "
        "email or phone number."
    ),
    ORDER: (
        "Retrieves order details: items, order status, shipping and tracking status, "
        "delivery address, estimated delivery date and refund eligibility. "
        "Accepts an order number (e.g. #12345), a carrier tracking number "
        "(e.g. 1Z999AA10123456784), or a customer email or phone number, in which case "
        "it returns that customer's most recent orders. "
        "Use it for questions about purchases, packages, shipping, delivery or refunds, "
        "e.g. 'Did my parcel go out yet?' or 'Can I still return order #24680?'. "
        "Do NOT use it for account-level questions such as account status, contact "
        "details or loyalty tier - use retrieve_customer_info for those."
    ),
}


# Stub data only: this exercise measures which tool Claude *selects*;
# the tool bodies are never executed by the harness.

def retrieve_customer_info(identifier: str):
    return f"Customer data for {identifier}"


def retrieve_order(identifier: str):
    return f"Order data for {identifier}"


def build_server(descriptions: dict[str, str]) -> MCPServer:
    """Build a server whose tools differ only in their descriptions."""
    server = MCPServer("DocumentMCP", log_level="ERROR")
    server.tool(name=CUSTOMER, description=descriptions[CUSTOMER])(retrieve_customer_info)
    server.tool(name=ORDER, description=descriptions[ORDER])(retrieve_order)
    return server


mcp = build_server(PRODUCTION)


if __name__ == "__main__":
    mcp.run(transport="stdio")
