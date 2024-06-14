import logging

import azure.functions as func
from llama_index.core.llms import ChatMessage

from laia.avatar import generate_video
from laia.rag import generate_response, hydrate_conversation
from laia.text2audio import generate_audio

logger = logging.getLogger(__name__)


app = func.FunctionApp()


@app.function_name(name="generate")
@app.route(route="generate", auth_level=func.AuthLevel.ANONYMOUS)
def generate_conversation(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f"Python HTTP trigger function processed a request: {req}")
    req_body = req.get_json()
    logger.info(f"Request body: {req_body}")
    try:
        messages = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in req_body["messages"]
        ]
        hydrate_conversation(messages, member_id=int(req_body["memberId"]))
        response = generate_response(messages)
        return func.HttpResponse(str(response), status_code=200)
    except Exception as e:
        logger.error(e)
        return func.HttpResponse("Internal Server Error", status_code=500)


@app.function_name(name="avatar")
@app.route(route="avatar", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def generate_avatar_video(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f"Python HTTP trigger function processed a request: {req}")
    req_body = req.get_json()
    logger.info(f"Request body: {req_body}")
    try:
        logger.info("Generating audio...")
        audio_path = generate_audio(text=req_body["message"])
        logger.info("Generating video...")
        video_path = generate_video(audio_path)
        return func.HttpResponse(
            body=video_path.read_bytes(), status_code=200, mimetype="video/mp4"
        )
    except Exception as e:
        logger.error(e)
        return func.HttpResponse("Internal Server Error", status_code=500)
