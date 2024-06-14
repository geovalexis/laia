import logging
import pathlib
from os import getenv

import azure.functions as func
from dotenv import load_dotenv
from llama_index.core import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.databricks import Databricks

logger = logging.getLogger(__name__)

load_dotenv()
app = func.FunctionApp()

# Define static data paths
INDEX_PERSIST_DIR = "../index"
DOCS_DATA_DIR = "../data"
PROMPTS_DIR = "../prompts"

# define LLM
llm = Databricks(
    model="databricks-meta-llama-3-70b-instruct",
    api_key=getenv("DATABRICKS_TOKEN"),
    api_base="https://adb-2978037251816793.13.azuredatabricks.net/serving-endpoints",
    max_tokens=256,
)
# build or load index
if pathlib.Path(INDEX_PERSIST_DIR).exists():
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
    index = load_index_from_storage(storage_context, llm=llm)
else:
    documents = SimpleDirectoryReader(DOCS_DATA_DIR).load_data()
    index = KeywordTableIndex.from_documents(documents, llm=llm)
    # Save the index for later use
    index.storage_context.persist(INDEX_PERSIST_DIR)

# Read system prompt
with open(pathlib.Path(PROMPTS_DIR) / "system.txt") as f:
    system_prompt = f.read()


# Create chat engine
def generate_response(messages: list[ChatMessage]):
    chat_engine = index.as_chat_engine(
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=10500),
        chat_mode=ChatMode.CONTEXT,
        system_prompt=system_prompt,
    )
    current_message = messages[-1]
    chat_history = messages[:-1]
    response = chat_engine.chat(str(current_message.content), chat_history=chat_history)
    return response


@app.function_name(name="generate")
@app.route(route="generate", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f"Python HTTP trigger function processed a request: {req}")
    req_body = req.get_json()
    logger.info(f"Request body: {req_body}")
    try:
        messages = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in req_body["messages"]
        ]
        response = generate_response(messages)
        return func.HttpResponse(str(response), status_code=200)
    except Exception as e:
        logger.error(e)
        return func.HttpResponse("Internal Server Error", status_code=500)
