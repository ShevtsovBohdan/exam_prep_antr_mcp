"""Tool distribution across a three-agent research system.

Each agent gets 4-5 tools scoped to its single responsibility. Tools are in the
Claude API `tools` format so they can be passed straight to messages.create().
"""


def tool(name, description, properties, required=None):
    """Build a tool definition in the Claude API format."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties),
        },
    }


def string(description):
    return {"type": "string", "description": description}


# Step 1: three roles, each with a scoped toolset. A tool belongs to exactly one
# role; if it could fit two, it is split or narrowed until it fits one.
AGENT_TOOLSETS = {
    "web_search": {
        "role": "Finds and retrieves content from the web. Does not analyse or write reports.",
        "first_tool_choice": {"type": "auto"},
        "tools": [
            tool(
                "search_web",
                "Searches the web for a query and returns ranked results (title, URL, snippet). "
                "Use to discover candidate sources; does not fetch full page content.",
                {"query": string("The search query")},
            ),
            tool(
                "fetch_page",
                "Fetches the full text content of a single web page by URL. "
                "Use after search_web to retrieve a promising result.",
                {"url": string("Absolute URL of the page to fetch")},
            ),
            tool(
                "extract_links",
                "Extracts all hyperlinks from a fetched web page. "
                "Use to follow references to further sources.",
                {"url": string("URL of a page already fetched with fetch_page")},
            ),
            tool(
                "save_snippet",
                "Saves a relevant text snippet together with its source URL so that "
                "downstream agents can analyse it. Returns a snippet ID.",
                {
                    "text": string("The snippet text, quoted verbatim"),
                    "source_url": string("URL the snippet was taken from"),
                },
            ),
        ],
    },
    "document_analysis": {
        "role": "Analyses the structure and content of retrieved documents. Does not search the web.",
        "first_tool_choice": {"type": "tool", "name": "extract_metadata"},
        "tools": [
            tool(
                "extract_metadata",
                "Extracts the title, author, publication date and document type of a source document.",
                {"document_id": string("ID of the saved document or snippet")},
            ),
            tool(
                "extract_data_points",
                "Extracts structured data fields (dates, amounts, names, version numbers) from a document.",
                {
                    "document_id": string("ID of the saved document or snippet"),
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Field types to extract, e.g. ['dates', 'versions']",
                    },
                },
            ),
            tool(
                "summarise_content",
                "Produces a concise summary of the key arguments in a single document.",
                {"document_id": string("ID of the saved document or snippet")},
            ),
            tool(
                "verify_claim",
                "Checks whether a specific claim is supported by the text of a source document. "
                "Returns supported / contradicted / not mentioned, with the relevant quote.",
                {
                    "claim": string("The claim to check"),
                    "document_id": string("ID of the document to check against"),
                },
            ),
            tool(
                "load_document",
                "Loads the text of a source document for analysis. Only accepts URLs on a "
                "trusted domain (modelcontextprotocol.io) that point to a document file "
                "(.pdf, .md, .txt or .html); any other URL is rejected. This is not a general "
                "web fetcher: if you need a page that does not meet these rules, do not retry "
                "with variations of the URL; report in your hand-off that the source could not "
                "be loaded.",
                {"url": string("Full URL of the document, e.g. https://modelcontextprotocol.io/specification/changelog.md")},
            ),
        ],
    },
    "synthesis": {
        "role": "Compiles analysed findings into a final report. Does not search or fetch sources.",
        "first_tool_choice": {"type": "auto"},
        "tools": [
            tool(
                "compile_report",
                "Assembles analysed findings into a structured report with sections and citations.",
                {
                    "title": string("Report title"),
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Analysed findings, each referencing a document ID",
                    },
                },
            ),
            tool(
                "format_citation",
                "Formats a source reference in the required citation style.",
                {
                    "document_id": string("ID of the source document"),
                    "style": {"type": "string", "enum": ["apa", "harvard", "ieee"]},
                },
            ),
            tool(
                "assess_coverage",
                "Evaluates whether every research question has been answered by the findings "
                "and lists any gaps.",
                {
                    "questions": {"type": "array", "items": {"type": "string"}},
                    "findings": {"type": "array", "items": {"type": "string"}},
                },
            ),
            tool(
                "verify_fact",
                "Quickly checks one simple factual claim (a date, version number, name or "
                "single statement) against ONE saved source document while compiling the report. "
                "Use it only when the answer can be found in a single document. "
                "Do NOT use it to compare several sources, resolve conflicting information or "
                "verify broad conclusions; for those, stop and state in your hand-off that the "
                "claim needs verification by the coordinator.",
                {
                    "claim": string("The fact to check"),
                    "document_id": string("ID of the one document to check against"),
                },
            ),
        ],
    },
}


def tool_names(agent):
    return [t["name"] for t in AGENT_TOOLSETS[agent]["tools"]]


def check_distribution(cross_role_tools=frozenset()):
    """Check the Step 1 acceptance criteria; returns a list of problems (empty = pass)."""
    problems = []
    owners = {}
    for agent in AGENT_TOOLSETS:
        names = tool_names(agent)
        if not 4 <= len(names) <= 5:
            problems.append(f"{agent} has {len(names)} tools (expected 4-5)")
        for name in names:
            if name in owners and name not in cross_role_tools:
                problems.append(f"{name} is shared by {owners[name]} and {agent}")
            owners.setdefault(name, agent)
    return problems


if __name__ == "__main__":
    for agent, config in AGENT_TOOLSETS.items():
        print(f"{agent} ({len(config['tools'])} tools): {', '.join(tool_names(agent))}")
    problems = check_distribution()
    print("\n" + ("\n".join(problems) if problems else "All distribution checks passed."))
