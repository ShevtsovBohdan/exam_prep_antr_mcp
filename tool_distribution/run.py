"""A coordinator running three real Claude agents, each with its scoped toolset.

Usage: uv run python tool_distribution/run.py ["your research question"]
"""

import sys

import anthropic

from agents import AGENT_TOOLSETS
from handlers import REPORTS, STORE, run_tool

# Sonnet 5 still accepts forced tool_choice ({"type": "tool"}), which Step 3
# needs; Sonnet 5.5 / Opus 5.5 reject it with a 400.
MODEL = "claude-sonnet-5"
MAX_TURNS = 12

# Every tool call made by any agent, in order: (agent, tool).
CALL_LOG: list[tuple[str, str]] = []


def run_agent(client, agent, task):
    """Run one agent's tool-use loop until it stops calling tools; returns its final text."""
    config = AGENT_TOOLSETS[agent]
    system = (
        f"You are the {agent.replace('_', ' ')} agent in a research team. {config['role']} "
        "Use only your own tools. When your part is done, reply with a short plain-text "
        "hand-off for the next agent, including any document IDs."
    )
    messages = [{"role": "user", "content": task}]

    choice=config["first_tool_choice"]
    for _ in range(MAX_TURNS):

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=config["tools"],
            tool_choice=choice,
            messages=messages,
        )
        choice = {"type": "auto"}
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in tool_uses:
            CALL_LOG.append((agent, block.name))
            print(f"  [{agent}] {block.name}({block.input})")
            text, is_error = run_tool(block.name, block.input)
            results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": text, "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    return f"[{agent} stopped after {MAX_TURNS} turns]"


def coordinate(question):
    """Coordinator: route the question through search -> analysis -> synthesis."""
    client = anthropic.Anthropic()

    print("\n== web_search ==")
    found = run_agent(client, "web_search", (
        f"Research question: {question}\n"
        "Find relevant sources, fetch them and save the key passages as snippets. "
        "Use at most 3 searches. As soon as you have fetched the relevant pages, save one "
        "snippet per page and hand off; do not keep looking for more sources."
    ))

    print("\n== document_analysis ==")
    analysed = run_agent(client, "document_analysis", (
        f"Research question: {question}\n"
        f"Saved documents: {sorted(STORE)}\n"
        f"Hand-off from web search: {found}\n"
        "Analyse each saved document and report the verified key findings."
    ))

    print("\n== synthesis ==")
    summary = run_agent(client, "synthesis", (
        f"Research question: {question}\n"
        f"Analysed findings: {analysed}\n"
        "Compile a cited report and check that the question is fully covered."
        "Analysed findings: ..."
        "Before including the release date in the report, confirm it against the source document.\n"
    ))
    return summary


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Research the latest MCP specification changes and compile a summary report"
    )
    final = coordinate(question)

    print("\n== report ==")
    print(REPORTS[-1] if REPORTS else final)

    print("\n== tool calls ==")
    for agent, tool in CALL_LOG:
        print(f"{agent:18} {tool}")
