from functools import cached_property
from typing import Optional

import tiktoken
from pydantic import BaseModel, computed_field

_encoding = tiktoken.get_encoding("cl100k_base")


class JobEvaluation(BaseModel):
    is_truly_remote: bool
    is_relevant_role: bool
    is_long_term: bool
    suitable: bool
    reasoning: str


class Job(BaseModel):
    title: str
    company: str
    description: str
    salary: Optional[str]
    date: Optional[str]
    location: Optional[str]
    link: str
    source: str                        # "remoteok" | "wwr" | "google"
    tags: Optional[list[str]]          # available from RemoteOK; None for others
    embedding: Optional[list[float]] = None
    similarity_score: Optional[float] = None
    llm_evaluation: Optional[JobEvaluation] = None

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @cached_property
    def embedding_text(self) -> str:
        parts = [f"Job Title: {self.title}"]
        if self.company:
            parts.append(f"Company: {self.company}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        parts.append(f"\n{self.description}")
        return "\n".join(parts)

    @computed_field
    @cached_property
    def token_count(self) -> int:
        return len(_encoding.encode(self.embedding_text))


def extract_job(job_dict: dict, source: str) -> Job:
    if source == "remoteok":
        salary_min = job_dict.get("salary_min", 0)
        salary_max = job_dict.get("salary_max", 0)
        salary = f"${salary_min}–${salary_max}" if (salary_min or salary_max) else None

        return Job(
            title=job_dict.get("position", ""),
            company=job_dict.get("company", ""),
            description=job_dict.get("description", ""),
            salary=salary,
            date=job_dict.get("date"),
            location=job_dict.get("location"),
            link=job_dict.get("url", ""),
            source="remoteok",
            tags=job_dict.get("tags"),
        )

    elif source == "wwr":
        raw_title = job_dict.get("title", "")
        if ": " in raw_title:
            company, title = raw_title.split(": ", 1)
        else:
            company, title = "", raw_title

        return Job(
            title=title,
            company=company,
            description=job_dict.get("description", ""),
            salary=None,
            date=job_dict.get("pubDate"),
            location=None,
            link=job_dict.get("link", ""),
            source="wwr",
            tags=None,
        )

    elif source == "google":
        apply_options = job_dict.get("apply_options") or []
        link = apply_options[0]["link"] if apply_options else job_dict.get("share_link", "")

        extensions = job_dict.get("extensions") or []
        salary = next(
            (e for e in extensions if any(c in e for c in ("$", "£", "€", "¥")) or "per year" in e.lower()),
            None,
        )

        return Job(
            title=job_dict.get("title", ""),
            company=job_dict.get("company_name", ""),
            description=job_dict.get("description", ""),
            salary=salary,
            date=None,
            location=job_dict.get("location"),
            link=link,
            source="google",
            tags=None,
        )

    else:
        raise ValueError(f"Unknown source: {source!r}. Expected 'remoteok', 'wwr', or 'google'.")
