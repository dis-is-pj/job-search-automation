from config import QUERIES, GLS, SIMILARITY_THRESHOLD
from fetchers import fetch_remoteok, fetch_wwr_rss_as_json, fetch_all_google_jobs
from models import extract_job
from scoring import embed, score_jobs
from evaluator import evaluate_jobs_with_llm
from sheets import push_jobs_to_sheet, add_to_csv


def run_pipeline():
    # 1. Fetch raw jobs from all sources
    print("--- Fetching jobs ---")
    try:
        remoteok_raw = fetch_remoteok()
        print(f"RemoteOK: {len(remoteok_raw)} jobs")
    except Exception as e:
        print(f"RemoteOK fetch failed: {e}")
        remoteok_raw = []

    try:
        wwr_raw = fetch_wwr_rss_as_json()
        print(f"WWR: {len(wwr_raw)} jobs")
    except Exception as e:
        print(f"WWR fetch failed: {e}")
        wwr_raw = []

    google_raw = fetch_all_google_jobs(QUERIES, GLS)
    print(f"Google Jobs: {len(google_raw)} total")

    # 2. Convert to unified Job objects
    print("\n--- Extracting to unified format ---")
    combined_jobs = []
    for job_dict in remoteok_raw:
        try:
            combined_jobs.append(extract_job(job_dict, "remoteok"))
        except Exception as e:
            print(f"  RemoteOK extract error: {e}")

    for job_dict in wwr_raw:
        try:
            combined_jobs.append(extract_job(job_dict, "wwr"))
        except Exception as e:
            print(f"  WWR extract error: {e}")

    for job_dict in google_raw:
        try:
            combined_jobs.append(extract_job(job_dict, "google"))
        except Exception as e:
            print(f"  Google extract error: {e}")

    print(f"Combined: {len(combined_jobs)} jobs")

    # 3. Deduplicate by URL
    seen = {}
    unique_jobs = []
    for job in combined_jobs:
        if job.link not in seen:
            seen[job.link] = job.source
            unique_jobs.append(job)
    print(f"After dedup: {len(unique_jobs)} unique jobs")

    # 4. Load profile and score
    print("\n--- Scoring ---")
    with open("profile_summary.txt") as f:
        profile_summary = f.read()
    profile_embedding = embed(profile_summary)
    score_jobs(unique_jobs, profile_embedding)

    # 5. Filter by similarity threshold
    filtered_jobs = [j for j in unique_jobs if (j.similarity_score or 0) > SIMILARITY_THRESHOLD]
    print(f"After similarity filter (>{SIMILARITY_THRESHOLD}): {len(filtered_jobs)} jobs")

    # 6. LLM evaluation
    print("\n--- LLM evaluation ---")
    evaluate_jobs_with_llm(filtered_jobs)

    # 7. Final list
    final_list = [j for j in filtered_jobs if j.llm_evaluation and j.llm_evaluation.suitable]
    print(f"Final suitable jobs: {len(final_list)}")

    # 8. Push results
    print("\n--- Pushing to sheets and CSV ---")
    push_jobs_to_sheet(final_list, sheet_name="final_list")
    push_jobs_to_sheet(filtered_jobs, sheet_name="shortlisted")
    add_to_csv(filtered_jobs)

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
