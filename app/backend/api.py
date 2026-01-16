from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from app.core.ai_agent import get_response_from_ai_agents
from app.config.settings import settings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="Multi AI Agent")

class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

@app.post("/chat")
def chat_endpoint(request: RequestState):
    logger.info(f"Received request for model : {request.model_name}")

    if request.model_name not in settings.ALLOWED_MODEL_NAMES:
        error_msg = f"Invalid model name: {request.model_name}. Allowed models: {settings.ALLOWED_MODEL_NAMES}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        response = get_response_from_ai_agents(
            request.model_name,
            request.messages,
            request.allow_search,
            request.system_prompt
        )

        logger.info(f"Successfully got response from AI agent {request.model_name}")

        return {"response": response}

    except Exception as e:
        error_detail = f"Error processing request for model '{request.model_name}': {str(e)}"
        logger.error(error_detail, exc_info=True)
        raise HTTPException(status_code=500, detail=error_detail)