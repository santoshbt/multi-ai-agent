from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


    ALLOWED_MODEL_NAMES = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b"
    ]

settings = Settings()