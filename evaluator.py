from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY
from models import Job, JobEvaluation

_client = OpenAI(api_key=OPENAI_API_KEY)

EVALUATOR_SYSTEM_PROMPT = """You are evaluating job postings to decide if they are suitable for a candidate.

Evaluate the job strictly against these three criteria:

1. REMOTE & LOCATION (is_truly_remote):
   - PASS if the job is worldwide remote, or if no location restriction is mentioned.
   - FAIL if the job says "remote" but restricts to a specific country, state, or city
     (e.g. "Remote - US only", "must reside in Canada", "Remote - NYC area").
   - Working from India must be possible. When in doubt, lean PASS.

2. ROLE RELEVANCE (is_relevant_role):
   - PASS for genuine technical AI/ML/Data Science roles: AI Engineer, ML Engineer,
     LLM Engineer, Applied Scientist, NLP Engineer, Data Scientist, MLOps, etc.
   - FAIL if the role is only tangentially related: AI data annotator, AI content
     reviewer, AI sales engineer, AI product manager with no hands-on technical work.

3. CONTRACT DURATION (is_long_term):
   - PASS if the role is permanent, long-term, or if duration is not mentioned.
   - FAIL only if explicitly stated as a short contract of 8 weeks or less
     (e.g. "6-week engagement", "2-month project").

Set overall suitable=True only if ALL three criteria pass.
Be concise in reasoning — one sentence per criterion is enough."""


def evaluate_single_job_llm(job: Job) -> JobEvaluation:
    response = _client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Job Title: {job.title}\nCompany: {job.company}\nLocation: {job.location or 'Not specified'}\n\n{job.description}"},
        ],
        response_format=JobEvaluation,
    )
    return response.choices[0].message.parsed


def evaluate_jobs_with_llm(jobs: list[Job], max_workers: int = 10) -> None:
    """Run LLM evaluation on all jobs in parallel, storing results in job.llm_evaluation in-place."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(evaluate_single_job_llm, job): job for job in jobs}
        for future in tqdm(as_completed(future_to_job), total=len(jobs)):
            job = future_to_job[future]
            job.llm_evaluation = future.result()
    print(f"Evaluated {len(jobs)} jobs with LLM.")
