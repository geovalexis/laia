import azure.functions as func
import logging
from os import getenv

from openai import OpenAI


logger = logging.getLogger(__name__)


app = func.FunctionApp()


@app.function_name(name="generate")
@app.route(route="generate", auth_level=func.AuthLevel.ANONYMOUS)
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f"Python HTTP trigger function processed a request: {req}")
    logger.info("DATABRICKS_TOKEN: %s", getenv("DATABRICKS_TOKEN"))
    try:
        client = OpenAI(
            api_key=getenv("DATABRICKS_TOKEN"),
            base_url="https://adb-2978037251816793.13.azuredatabricks.net/serving-endpoints",
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": "Tell me about Large Language Models"},
            ],
            model="databricks-meta-llama-3-70b-instruct",
            max_tokens=256,
        )
        logger.info(chat_completion)
        return_value = "Mocked response from llama3"
        return func.HttpResponse(return_value, status_code=200)
    except Exception as e:
        logger.error(e)
        return func.HttpResponse("Internal Server Error", status_code=500)
