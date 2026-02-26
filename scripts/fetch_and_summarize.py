#!/usr/bin/env python3
"""
Fetch today's arxiv and HuggingFace submissions and summarize them using AI.
"""

import os
import sys
import json
import yaml
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateutil_parser
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
HF_PAPERS_API = "https://huggingface.co/api/daily_papers"
MAX_ARXIV_RESULTS = 100
MAX_HF_RESULTS = 50
SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "..", "summaries")


def load_topics(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── arXiv fetching ─────────────────────────────────────────────────────────────

def fetch_arxiv_papers(categories: list[str], keywords: list[str]) -> list[dict]:
    """Fetch today's arxiv papers matching the given categories and keywords."""
    # Build category query
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_query})"

    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": MAX_ARXIV_RESULTS,
    }

    resp = requests.get(ARXIV_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    feed = feedparser.parse(resp.content)

    # Keep only papers submitted within the last 2 days (arxiv publishes in batches)
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    lower_keywords = [kw.lower() for kw in keywords]

    papers = []
    for entry in feed.entries:
        published = dateutil_parser.parse(entry.get("published", ""))
        if published < cutoff:
            continue

        title = entry.get("title", "").replace("\n", " ").strip()
        abstract = entry.get("summary", "").replace("\n", " ").strip()
        combined = (title + " " + abstract).lower()

        # Filter by keywords (optional: skip if no keywords match to stay focused)
        if keywords and not any(kw in combined for kw in lower_keywords):
            continue

        authors = [a.get("name", "") for a in entry.get("authors", [])]
        link = entry.get("link", "")
        arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else link

        papers.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "link": link,
            "published": published.strftime("%Y-%m-%d"),
        })

    return papers


# ── HuggingFace fetching ───────────────────────────────────────────────────────

def fetch_hf_papers(keywords: list[str]) -> list[dict]:
    """Fetch today's HuggingFace daily papers matching keywords."""
    resp = requests.get(HF_PAPERS_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    lower_keywords = [kw.lower() for kw in keywords]
    papers = []
    for item in data[:MAX_HF_RESULTS]:
        paper = item.get("paper", {})
        title = paper.get("title", "").replace("\n", " ").strip()
        abstract = paper.get("summary", "").replace("\n", " ").strip()
        combined = (title + " " + abstract).lower()

        if keywords and not any(kw in combined for kw in lower_keywords):
            continue

        arxiv_id = paper.get("id", "")
        link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else item.get("url", "")
        published_at = item.get("publishedAt", "")
        date_str = published_at[:10] if published_at else ""

        authors = [a.get("name", "") for a in paper.get("authors", [])]

        papers.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "link": link,
            "published": date_str,
            "upvotes": item.get("upvotes", 0),
        })

    return papers


# ── AI summarization ───────────────────────────────────────────────────────────

def build_papers_text(papers: list[dict], max_papers: int = 20) -> str:
    """Format papers into a compact text block for the LLM prompt."""
    lines = []
    for i, p in enumerate(papers[:max_papers], 1):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += " et al."
        abstract_short = p["abstract"][:300] + "..." if len(p["abstract"]) > 300 else p["abstract"]
        lines.append(
            f"{i}. **{p['title']}**\n"
            f"   Authors: {authors_str}\n"
            f"   Link: {p['link']}\n"
            f"   Abstract: {abstract_short}"
        )
    return "\n\n".join(lines)


def summarize_with_ai(client: OpenAI, arxiv_papers: list[dict], hf_papers: list[dict], date_str: str) -> str:
    """Use OpenAI to produce a structured Markdown summary."""
    arxiv_text = build_papers_text(arxiv_papers) if arxiv_papers else "No matching papers found today."
    hf_text = build_papers_text(hf_papers) if hf_papers else "No matching papers found today."

    system_prompt = (
        "You are a research assistant helping a machine learning researcher stay up-to-date. "
        "Given a list of papers from arXiv and HuggingFace, produce a concise, well-structured "
        "Markdown daily digest. For each paper write a 2-3 sentence plain-English summary. "
        "Group the papers by theme if possible. Be informative but brief."
    )

    user_prompt = f"""Today is {date_str}. Please summarize the following papers for a daily digest.

## arXiv Papers

{arxiv_text}

## HuggingFace Daily Papers

{hf_text}

Produce a Markdown document with:
1. A brief overall summary (3-5 sentences) of the day's research trends.
2. Individual paper summaries organized under "### arXiv Highlights" and "### HuggingFace Highlights" sections.
   For each paper include: title as a link, a 2-3 sentence summary, and key contributions.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    return response.choices[0].message.content


# ── Output ─────────────────────────────────────────────────────────────────────

def save_summary(summary: str, date_str: str, arxiv_count: int, hf_count: int) -> str:
    """Write the Markdown summary to summaries/<date>.md and return the path."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    filepath = os.path.join(SUMMARY_DIR, f"{date_str}.md")

    header = (
        f"# Daily Research Digest — {date_str}\n\n"
        f"*Fetched {arxiv_count} arXiv paper(s) and {hf_count} HuggingFace paper(s) matching configured topics.*\n\n"
        "---\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + summary)

    return filepath


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "topics.yml")
    topics = load_topics(config_path)

    arxiv_cfg = topics.get("arxiv", {})
    hf_cfg = topics.get("huggingface", {})

    categories = arxiv_cfg.get("categories", [])
    arxiv_keywords = arxiv_cfg.get("keywords", [])
    hf_keywords = hf_cfg.get("keywords", [])

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[{date_str}] Fetching arXiv papers (categories: {categories})...")
    arxiv_papers = fetch_arxiv_papers(categories, arxiv_keywords)
    print(f"  → Found {len(arxiv_papers)} matching arXiv papers.")

    print(f"[{date_str}] Fetching HuggingFace daily papers...")
    hf_papers = fetch_hf_papers(hf_keywords)
    print(f"  → Found {len(hf_papers)} matching HuggingFace papers.")

    if not arxiv_papers and not hf_papers:
        print("No papers found for today. Skipping summary generation.")
        return

    print("Generating AI summary...")
    client = OpenAI(api_key=api_key)
    summary = summarize_with_ai(client, arxiv_papers, hf_papers, date_str)

    filepath = save_summary(summary, date_str, len(arxiv_papers), len(hf_papers))
    print(f"Summary saved to: {filepath}")


if __name__ == "__main__":
    main()
