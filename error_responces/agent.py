import asyncio
import json

import anthropic
from anthropic.types import Message
from mcp import Client

from server import mcp

MODEL = "claude-sonnet-5"
MAX_TURNS = 10
MAX_RETRIES = 3  # transient retries after the first attempt, backing off 1s, 2s, 4s

SYSTEM = """You are a customer-support agent. Recover from tool errors using the error metadata:
- transient (isRetryable: true): call the same tool again with the same arguments.
- validation: fix the input as the description says, then call again.
- business: do not retry; call escalate_to_human with escalation_type "manager_approval".
- permission: do not retry; call escalate_to_human with escalation_type "elevated_credentials".
- A successful result with resultCount 0 is a valid answer, not an error: report it and do not retry.

failure_mode is a test switch. Use the value the user gives on the first call. On retries or corrected
calls, use failure_mode "success" unless the user says the failure is persistent."""

ESCALATE_TOOL = {
    "name": "escalate_to_human",
    "description": (
        "Hands the case to a human when the agent cannot resolve it: manager_approval for "
        "business-rule limits, elevated_credentials for permission errors."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "escalation_type": {"type": "string", "enum": ["manager_approval", "elevated_credentials"]},
            "reason": {"type": "string", "description": "What failed and what the human needs to do."},
        },
        "required": ["escalation_type", "reason"],
    },
}


async def claude_tools(client):
    """Convert the MCP server's tool list into the Claude API `tools` format."""
    mcp_tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in (await client.list_tools()).tools
    ]
    return mcp_tools + [ESCALATE_TOOL]


def add_user_message(messages, message):
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(assistant_message)


def tool_result(tool_use_id, text, is_error):
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
        "is_error": is_error,
    }


async def run_tools(mcp_client, tool_requests, state):
    """Run every tool_use block. The retry cap and backoff are enforced here, not left to the model."""
    tool_results = []

    for tool_request in tool_requests:
        if tool_request.name == "escalate_to_human":
            print(f"  ESCALATED ({tool_request.input['escalation_type']}): {tool_request.input['reason']}")
            tool_results.append(tool_result(tool_request.id, "Escalation ticket created.", False))
            continue

        failures = state["transient_failures"]
        if failures > MAX_RETRIES:
            print("  BLOCKED: retry limit reached, call not executed")
            tool_results.append(tool_result(
                tool_request.id,
                json.dumps({
                    "errorCategory": "transient",
                    "isRetryable": False,
                    "description": f"Retry limit ({MAX_RETRIES}) reached. Do not retry; tell the user "
                                   "the customer database is unavailable.",
                }),
                True,
            ))
            continue
        if failures:
            delay = 2 ** (failures - 1)
            print(f"  backoff {delay}s before retry {failures}/{MAX_RETRIES}")
            await asyncio.sleep(delay)

        print(f"  call {tool_request.name}({tool_request.input})")
        result = await mcp_client.call_tool(tool_request.name, tool_request.input)
        text = "\n".join(b.text for b in result.content)
        print(f"    -> is_error={result.is_error} {text[:100]}")

        if result.is_error and json.loads(text).get("errorCategory") == "transient":
            state["transient_failures"] += 1
        else:
            state["transient_failures"] = 0

        tool_results.append(tool_result(tool_request.id, text, result.is_error))

    return tool_results


async def invoke_tool(anthropic_client, query, system=SYSTEM):
    """Run the agent loop until Claude stops calling tools; return the final response."""
    messages = [{"role": "user", "content": query}]
    state = {"transient_failures": 0}

    async with Client(mcp) as mcp_client:
        tools = await claude_tools(mcp_client)
        for _ in range(MAX_TURNS):
            response = await anthropic_client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=tools,
                messages=messages,
            )
            add_assistant_message(messages, response)
            if response.stop_reason != "tool_use":
                return response

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            add_user_message(messages, await run_tools(mcp_client, tool_uses, state))

    raise RuntimeError(f"Agent did not finish within {MAX_TURNS} turns")


SCENARIOS = {
    "success": "Look up customer CUST-10042 with failure_mode=success.",
    "not_found": "Look up customer CUST-99999 with failure_mode=not_found.",
    "transient (recovers)": "Look up customer CUST-10042 with failure_mode=transient.",
    "transient (persistent)": "Look up customer CUST-10042 with failure_mode=transient. "
                              "The outage is persistent: keep failure_mode=transient on every attempt.",
    "validation": "Look up customer CUST10042 with failure_mode=validation.",
    "business": "Process a £750 refund for CUST-10042: look them up with failure_mode=business.",
    "permission": "Show financial records for CUST-10042: look them up with failure_mode=permission.",
}


async def main():
    client = anthropic.AsyncAnthropic()
    for name, query in SCENARIOS.items():
        print(f"\n=== {name}")
        response = await invoke_tool(client, query)
        final = " ".join(b.text for b in response.content if b.type == "text")
        print(f"  CLAUDE: {final[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
