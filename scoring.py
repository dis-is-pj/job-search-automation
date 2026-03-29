import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)


def embed(text: str) -> list[float]:
    response = _client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def embed_jobs(jobs: list, batch_size: int = 500) -> None:
    """Embed all jobs in batches, storing the result in job.embedding in-place."""
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        response = _client.embeddings.create(
            input=[job.embedding_text for job in batch],
            model=EMBEDDING_MODEL,
        )
        for job, result in zip(batch, sorted(response.data, key=lambda x: x.index)):
            job.embedding = result.embedding
        print(f"Embedded {min(i + batch_size, len(jobs))}/{len(jobs)} jobs")


def score_jobs(jobs: list, profile_embedding: list[float]) -> None:
    """Embed all jobs then compute cosine similarity against profile_embedding in-place.
    OpenAI embeddings are unit-normalized, so cosine similarity == dot product.
    """
    embed_jobs(jobs)
    profile_vec = np.array(profile_embedding)
    job_vecs = np.array([job.embedding for job in jobs])
    scores = job_vecs @ profile_vec
    for job, score in zip(jobs, scores):
        job.similarity_score = float(score)
    print("Scored all jobs against profile embedding.")
