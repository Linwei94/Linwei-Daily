# Linwei-Daily

A daily AI-powered digest of arXiv and HuggingFace research submissions on topics Linwei is interested in.

## How it works

1. Every day at 09:00 UTC a GitHub Actions workflow runs `scripts/fetch_and_summarize.py`.
2. The script fetches recent papers from **arXiv** (via the arXiv API) and **HuggingFace** (via the HuggingFace daily-papers API), filtering by the categories and keywords defined in `config/topics.yml`.
3. The collected papers are sent to **OpenAI GPT-4o-mini**, which produces a structured Markdown digest.
4. The resulting summary is committed to `summaries/YYYY-MM-DD.md`.

## Setup

1. Fork this repository.
2. Add an `OPENAI_API_KEY` secret in *Settings → Secrets and variables → Actions*.
3. Edit `config/topics.yml` to match the topics you care about.
4. The workflow will run automatically every day, or you can trigger it manually from the *Actions* tab.

## Repository structure

```
config/
  topics.yml              # arXiv categories & keywords to monitor
scripts/
  fetch_and_summarize.py  # Main script
summaries/
  YYYY-MM-DD.md           # Generated daily digests
.github/workflows/
  daily_summary.yml       # Scheduled GitHub Actions workflow
requirements.txt          # Python dependencies
```
