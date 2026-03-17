import csv
import os
from datetime import date
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv(override=True)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_COLUMNS = [
    "Date Added",
    "Job Title",
    "Company",
    "URL",
    "Source",
    "Similarity Score",
    "Salary",
    "Location",
    "Applied?",
    "Notes",
]


def _get_worksheet(sheet_name: str):
    creds = Credentials.from_service_account_file(
        os.getenv("GOOGLE_CREDENTIALS_PATH"), scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
    return sheet.worksheet(sheet_name)


def _get_existing_urls(worksheet) -> set[str]:
    """Fetch all URLs already in the sheet (column 4) to avoid duplicates."""
    url_col = worksheet.col_values(4)  # 1-indexed; col 4 = URL
    return set(url_col[1:])            # skip header row


def push_jobs_to_sheet(jobs, sheet_name: str = "final_list") -> None:
    """Append jobs to the named sheet, skipping any already present by URL."""
    worksheet = _get_worksheet(sheet_name)
    existing_urls = _get_existing_urls(worksheet)

    today = date.today().strftime("%Y-%m-%d")
    rows_to_append = []

    for job in jobs:
        if job.link in existing_urls:
            continue

        evaluation = job.llm_evaluation
        rows_to_append.append([
            today,
            job.title,
            job.company,
            job.link,
            job.source,
            round(job.similarity_score, 4) if job.similarity_score is not None else "",
            job.salary or "",
            job.location or "",
            evaluation.suitable if evaluation else "",
            evaluation.reasoning if evaluation else "",
            "",   # Applied? — manual
            "",   # Notes — manual
        ])

    if not rows_to_append:
        print("No new jobs to add — all already in the sheet.")
        return

    worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
    print(f"Added {len(rows_to_append)} new jobs to the sheet. ({len(jobs) - len(rows_to_append)} duplicates skipped)")


def add_to_csv(jobs, filepath: str = "jobs_analysis.csv") -> None:
    """Append jobs to a local CSV file, skipping duplicates by URL.
    All fields from llm_evaluation are included dynamically — stays in sync
    with whatever fields are defined on JobEvaluation.
    """
    path = Path(filepath)
    file_exists = path.exists()

    # Read existing URLs to dedup
    existing_urls = set()
    if file_exists:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_urls = {row["URL"] for row in reader}

    today = date.today().strftime("%Y-%m-%d")

    # Derive llm_evaluation column names from the first job that has one
    llm_fields = []
    for job in jobs:
        if job.llm_evaluation is not None:
            llm_fields = [f"llm_{k}" for k in job.llm_evaluation.model_dump().keys()]
            break

    base_fields = [
        "Date Added", "Job Title", "Company", "URL", "Source",
        "Similarity Score", "Salary", "Location",
    ]
    fieldnames = base_fields + llm_fields

    new_rows = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for job in jobs:
            if job.link in existing_urls:
                continue

            row = {
                "Date Added": today,
                "Job Title": job.title,
                "Company": job.company,
                "URL": job.link,
                "Source": job.source,
                "Similarity Score": round(job.similarity_score, 4) if job.similarity_score is not None else "",
                "Salary": job.salary or "",
                "Location": job.location or "",
            }

            if job.llm_evaluation is not None:
                for k, v in job.llm_evaluation.model_dump().items():
                    row[f"llm_{k}"] = v

            writer.writerow(row)
            new_rows += 1

    print(f"Added {new_rows} jobs to {path}. ({len(jobs) - new_rows} duplicates skipped)")
