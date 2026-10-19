"""Simulated tool implementations.

The agents and the model calls are real; the "web" and "documents" are canned
so runs are deterministic and nothing leaves the machine except API calls.
"""

import json
from urllib.parse import urlparse
from pathlib import PurePosixPath

FAKE_WEB = {
    "https://modelcontextprotocol.io/specification/changelog.md": {
        "title": "MCP Specification Changelog",
        "author": "MCP Working Group",
        "date": "2025-06-18",
        "text": (
            "Version 2025-06-18 of the Model Context Protocol specification introduces "
            "structured tool output, removes JSON-RPC batching, requires the "
            "MCP-Protocol-Version header on HTTP requests, and adds elicitation so servers "
            "can request additional information from users during a session."
        ),
        "links": ["https://modelcontextprotocol.io/specification/2025-06-18/server/tools.md"],
    },
    "https://modelcontextprotocol.io/specification/2025-06-18/server/tools.md": {
        "title": "MCP Tools",
        "author": "MCP Working Group",
        "date": "2025-06-18",
        "text": (
            "Tools may now declare an outputSchema. Servers returning structured content "
            "must conform to it. Tool results may include resource links."
        ),
        "links": [],
    },
}

TRUSTED_DOMAINS = ["modelcontextprotocol.io"]
DOCUMENT_EXTENSIONS = [".pdf", ".md", ".txt", ".html"]

# Snippets saved by the web search agent, keyed by snippet ID. This is the
# hand-off point between agents: later agents only ever see IDs.
STORE: dict[str, dict] = {}
REPORTS: list[str] = []


def _page(url):
    page = FAKE_WEB.get(url)
    if page is None:
        raise ValueError(f"404: {url} not found. Do not guess URLs; only fetch URLs returned by search_web or extract_links.")
    return page


def _doc(document_id):
    doc = STORE.get(document_id)
    if doc is None:
        raise ValueError(f"Unknown document_id {document_id!r}. Known: {sorted(STORE)}")
    return doc


# --- web search -------------------------------------------------------------

def search_web(query):
    return [
        {"title": p["title"], "url": url, "snippet": p["text"][:100] + "..."}
        for url, p in FAKE_WEB.items()
    ]


def fetch_page(url):
    return _page(url)["text"]


def extract_links(url):
    return _page(url)["links"]

def verify_fact(claim, document_id):
    return verify_claim(claim, document_id)

def save_snippet(text, source_url):
    snippet_id = f"doc-{len(STORE) + 1}"
    STORE[snippet_id] = {"text": text, "source_url": source_url}
    return {"snippet_id": snippet_id}


# --- document analysis ------------------------------------------------------

def extract_metadata(document_id):
    doc = _doc(document_id)
    page = FAKE_WEB.get(doc["source_url"], {})
    return {
        "title": page.get("title", "Unknown"),
        "author": page.get("author", "Unknown"),
        "date": page.get("date", "Unknown"),
        "type": "web page",
        "source_url": doc["source_url"],
    }


def extract_data_points(document_id, fields):
    # Returning the raw text and letting the model pull out the fields keeps the
    # stub simple; a real implementation would parse.
    return {"fields_requested": fields, "text": _doc(document_id)["text"]}


def summarise_content(document_id):
    return {"text_to_summarise": _doc(document_id)["text"]}


def verify_claim(claim, document_id):
    text = _doc(document_id)["text"].lower()
    words = [w for w in claim.lower().split() if len(w) > 3]
    hits = sum(w in text for w in words)
    status = "supported" if words and hits / len(words) >= 0.6 else "not mentioned"
    return {"status": status, "source_text": _doc(document_id)["text"]}

def load_document(url):
    parsed = urlparse(url)
    extension = PurePosixPath(parsed.path).suffix

    if parsed.hostname not in TRUSTED_DOMAINS:
        raise ValueError(f"Rejected: {url} is not trusted")
    
    if extension not in DOCUMENT_EXTENSIONS:
        raise ValueError(f"Rejected: {extension} is not allowed")

    return _page(url)["text"]

# --- synthesis --------------------------------------------------------------

def compile_report(title, findings):
    report = f"# {title}\n\n" + "\n".join(f"- {f}" for f in findings)
    REPORTS.append(report)
    return {"status": "compiled", "report": report}


def format_citation(document_id, style):
    meta = extract_metadata(document_id)
    return f"{meta['author']} ({meta['date'][:4]}). {meta['title']}. {meta['source_url']}"


def assess_coverage(questions, findings):
    return {"questions": questions, "finding_count": len(findings)}


HANDLERS = {f.__name__: f for f in [
    search_web, fetch_page, extract_links, save_snippet,
    extract_metadata, extract_data_points, summarise_content, verify_claim,
    compile_report, format_citation, assess_coverage, verify_fact, load_document
]}


def run_tool(name, tool_input):
    """Execute a tool by name; returns (result_text, is_error)."""
    try:
        return json.dumps(HANDLERS[name](**tool_input)), False
    except Exception as e:  # surfaced to the model as an error tool_result
        return f"{type(e).__name__}: {e}", True
