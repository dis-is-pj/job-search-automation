import os
from dotenv import load_dotenv

load_dotenv(override=True)

SERPAPI_API_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.45

QUERIES = [
    'remote ("AI Engineer" OR "Machine Learning Engineer" OR "Applied AI Engineer")',
    'remote ("LLM" OR "Large Language Model" OR "Generative AI" OR "GenAI" OR "NLP")',
    'remote ("LangChain" OR "LlamaIndex" OR "RAG" OR "OpenAI API" OR "GPT")',
    'remote ("LLM deployment" OR "model serving" OR "inference" OR "ML Platform" OR "MLOps")',
]

GLS = ["us", "ca", "gb", "de", "nl", "sg", "au"]
