import logging

import azure.functions as func
from llama_index.core.llms import ChatMessage

from laia.llamaindex import generate_response, hydrate_conversation

logger = logging.getLogger(__name__)


app = func.FunctionApp()


@app.function_name(name="generate")
@app.route(route="generate", auth_level=func.AuthLevel.ANONYMOUS)
def test_function(req: func.HttpRequest) -> func.HttpResponse:
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
