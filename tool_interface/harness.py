import asyncio

import anthropic

from server import CUSTOMER, ORDER, PRODUCTION, VAGUE, build_server

MODEL = "claude-sonnet-5"

# Step 5: a realistic-sounding support prompt whose wording ("check customer
# details", "verify the account", "look up the customer") echoes the
# retrieve_customer_info description and applies to *every* request.
CONFLICTING = (
    "You are a support assistant for an online shop. "
    "Always check customer details before proceeding with any request. "
    "Verify the customer's account first whenever they contact us. "
    "Look up the customer by their email or phone number before answering anything."
)

# Same business intent, but conditional and deferring to the tool descriptions.
FIXED = (
    "You are a support assistant for an online shop. "
    "Pick the tool whose description best matches what the user is asking about. "
    "Questions about orders, deliveries or refunds can be answered from an order "
    "number, tracking number, email or phone number - no separate account lookup is needed. "
    "Only fetch the account profile when the question is about the account itself."
)

# (query, expected tool). The last five are traps: customer-flavoured wording
# or identifiers attached to an order intent, and vice versa.
QUERIES: list[tuple[str, str]] = [
    ("What is the status of order #12345?", ORDER),
    ("Look up customer john@example.com", CUSTOMER),
    ("Find the account for phone 555-0123", CUSTOMER),
    ("Is order #67890 eligible for a refund?", ORDER),
    ("What loyalty tier is customer C-1042 on?", CUSTOMER),
    ("Where is my package? My email is jane@example.com", ORDER),
    ("Has john@example.com's last purchase shipped yet?", ORDER),
    ("What address will order #11111 be delivered to?", ORDER),
    ("Is the account for jane@example.com still active?", CUSTOMER),
    ("Tracking 1Z999AA10123456784, when will it arrive?", ORDER),
]


def claude_tools(descriptions):
    """Convert the MCP server's tool list into the Claude API `tools` format."""
    server = build_server(descriptions)
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in asyncio.run(server.list_tools())
    ]


def pick_tool(client, tools, query, system=None):
    """Return the name of the tool Claude chose, or None if it answered in text."""
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": query}],
        **kwargs,
    )
    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    return tool_use.name if tool_use else None


def short(tool):
    return {CUSTOMER: "customer", ORDER: "order"}.get(tool, "(no tool)")


def compare(client, system=None):
    """Run every query against the vague and production descriptions; print a table."""
    before, after = claude_tools(VAGUE), claude_tools(PRODUCTION)
    score = {"before": 0, "after": 0}
    print(f"{'expected':<9} | {'before':<15} | {'after':<15} | query")
    for query, expected in QUERIES:
        cells = []
        for label, tools in (("before", before), ("after", after)):
            picked = pick_tool(client, tools, query, system)
            ok = picked == expected
            score[label] += ok
            cells.append(f"{'OK ' if ok else 'BAD'} {short(picked)}")
        print(f"{short(expected):<9} | {cells[0]:<15} | {cells[1]:<15} | {query}")
    n = len(QUERIES)
    print(f"Accuracy  before: {score['before']}/{n}   after: {score['after']}/{n}")


if __name__ == "__main__":
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    print("=== no system prompt")
    compare(client)
    print("\n=== conflicting system prompt")
    compare(client, system=CONFLICTING)
    print("\n=== fixed system prompt")
    compare(client, system=FIXED)
