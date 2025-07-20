This release brings major bug fixes, performance improvements, and new capabilities to the original Deep Research at Home function.
🛠️ Stability & Logic Fixes
✅ JSON Parsing Crashes Resolved

Malformed or incomplete JSON from LLMs used to crash the system. Now, a repair function tries multiple fallback strategies:

    Extract JSON from Markdown blocks

    Fix broken syntax with regex

    Ask a smaller model to reconstruct it
    If all fail, a fallback outline is generated, and the UI informs the user that recovery was attempted.

✅ Fixed Resource Leaks & Backend Freezes

Thread pools previously stayed alive after task failure/completion, causing memory leaks and freezes. Now, threads are properly cleaned up after each research task.
✅ Numerical Computation Safeguards

NaN, Inf, or dimension mismatches in NumPy vector operations are now detected and avoided before any computation occurs.
✅ Safer Key & Index Access

All dict and list accesses now use .get() with fallbacks or include length checks to prevent key/index exceptions.
✅ Upgraded Caching System

Replaced the old cache with:

    An LRU (Least Recently Used) eviction strategy

    SHA256-based cache key hashing
    Improves consistency and avoids collisions.

✅ Semantic Logic Corrections

    Fixed incorrect variable references during content compression

    Rewrote the topic-flattening logic to fully support deeply nested research outlines

✅ Cleaner HTML Parsing

Replaced fragile regex parsing with BeautifulSoup. Irrelevant HTML elements (scripts, nav bars, etc.) are stripped. Final output is cleaner and more readable.
✅ Removed Ollama Token Counting Dependency

Previously relied on a remote ollama/tokenize endpoint. This has been removed and replaced with a local, approximate token estimation method based on word count.
✅ Interactive Output

Research results are now displayed in collapsible sections:

    High-level summaries first

    Detailed sources on-demand
    This keeps the UI clean and makes long outputs easier to scan.

🔍 SearXNG Setup for Plugin Search (Docker Version)

SearXNG is a privacy-respecting, self-hosted metasearch engine. It's perfect for Open WebUI plugins — no API key, no rate limits, no tracking.
🚀 Quick Setup (Assumes Docker is Installed)
1. Clone the SearXNG Docker Repo

git clone https://github.com/searxng/searxng-docker.git
cd searxng-docker

2. Edit .env

Ensure the following:

SEARXNG_HOSTNAME=localhost:8080

    ⚠️ Do not include http:// or a trailing slash.

3. Edit docker-compose.yaml

Ensure the searxng service block includes:

services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://${SEARXNG_HOSTNAME:-localhost}

    ✅ Must use http:// — do not use https://
    ✅ Ensure nothing else (like caddy) is binding to port 8080

4. Start the Container

docker compose up -d

5. Verify It’s Running

Open in browser:

http://localhost:8080

Or test from terminal:

curl http://localhost:8080/search?q=test&format=json

🔌 Plugin Configuration

If your plugin requires a search URL, use:

http://host.docker.internal:8080/search?q=

    Works inside Docker containers (like Open WebUI) on Windows/macOS

    For Linux, use:

http://172.17.0.1:8080/search?q=

💡 Optional Enhancements
Use port 8888 instead:

ports:
  - "8888:8080"

Then access via http://host.docker.internal:8888/search?q=
Enable JSON Format:

After first launch, open:

searxng/settings.yml

And add:

formats: ["html", "json"]

✅ Plugin Integration Example

{
  "search_url": "http://host.docker.internal:8080/search?q=",
  "use_format_json": true
}

💬 Final Notes

Your updated plugin is now backed by a robust, local, privacy-friendly search engine with crash-resilience and high performance.
No external APIs. No instability. Just reliable research — powered entirely by you.

And of course, big shourout to @radeon for making the original deep research at home pipe on openwebui, all I did is fixed bugs for him and did some improvements in my openion.
