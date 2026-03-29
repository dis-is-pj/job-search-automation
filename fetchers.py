import html
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from config import SERPAPI_API_KEY


def _html_to_clean_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    decoded = html.unescape(raw_html)
    soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_remoteok(timeout=30) -> list[dict]:
    url = "https://remoteok.com/api"
    headers = {
        "User-Agent": "pj-job-fetcher/0.1 (+https://example.local)",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = []
    for job in r.json():
        if "legal" in job:
            continue
        job["description_html"] = job.get("description", "")
        job["description"] = _html_to_clean_text(job["description_html"])
        data.append(job)
    return data


def fetch_wwr_rss_as_json(timeout=30) -> list[dict]:
    rss_url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
    r = requests.get(rss_url, headers={"User-Agent": "pj-job-fetcher/0.1"}, timeout=timeout)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        items.append({
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "guid": (item.findtext("guid") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
            "description_html": (item.findtext("description") or "").strip(),
            "description": _html_to_clean_text(item.findtext("description") or ""),
        })
    return items


def fetch_serpapi_google_jobs(
    query: str,
    gl: str | list,
    max_pages: int = 5,
    timeout: int = 30,
    retry: int = 1,
) -> list[dict]:
    if not SERPAPI_API_KEY:
        raise RuntimeError("Missing SERPAPI_KEY env var.")
    if max_pages < 1:
        return []

    gl_list = [gl] if isinstance(gl, str) else list(gl)
    base = "https://serpapi.com/search.json"
    headers = {"User-Agent": "pj-job-fetcher/0.1", "Accept": "application/json"}
    all_jobs = []

    for gl_ in gl_list:
        params = {
            "engine": "google_jobs",
            "q": query,
            "hl": "en",
            "gl": gl_,
            "api_key": SERPAPI_API_KEY,
        }
        next_page_token = None
        seen_tokens: set[str] = set()

        for _ in range(1, max_pages + 1):
            if next_page_token:
                if next_page_token in seen_tokens:
                    break
                seen_tokens.add(next_page_token)
                params["next_page_token"] = next_page_token
            else:
                params.pop("next_page_token", None)

            url = f"{base}?{urllib.parse.urlencode(params)}"
            last_err = None
            for attempt in range(retry + 1):
                try:
                    r = requests.get(url, headers=headers, timeout=timeout)
                    r.raise_for_status()
                    data = r.json()
                    last_err = None
                    break
                except requests.HTTPError as e:
                    last_err = e
                    if attempt < retry and getattr(e.response, "status_code", None) in (429, 500, 502, 503, 504):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise
            if last_err:
                raise last_err

            jobs = data.get("jobs_results") or []
            for j in jobs:
                if isinstance(j, dict):
                    j["_gl"] = gl_
                    j["_query"] = query
            all_jobs.extend(jobs)

            next_page_token = (data.get("serpapi_pagination") or {}).get("next_page_token")
            if not next_page_token:
                break

    return all_jobs


def fetch_all_google_jobs(queries: list[str], gls: list[str], max_pages: int = 5) -> list[dict]:
    all_jobs = []
    for query in queries:
        try:
            jobs = fetch_serpapi_google_jobs(query=query, gl=gls, max_pages=max_pages)
            all_jobs.extend(jobs)
            print(f"Google Jobs: fetched {len(jobs)} for '{query[:60]}'")
        except Exception as e:
            print(f"Error fetching Google Jobs for query '{query[:60]}': {e}")
    return all_jobs
