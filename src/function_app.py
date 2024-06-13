import azure.functions as func
import logging
from openai import OpenAI
from os import getenv
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

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
        return func.HttpResponse(
            chat_completion.choices[0].message.content, status_code=200
        )
    except Exception as e:
        logger.error(e)
        return func.HttpResponse("Internal Server Error", status_code=500)
