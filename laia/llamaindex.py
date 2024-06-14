import pathlib
from os import getenv

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    TreeIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.databricks import Databricks

load_dotenv()

# Define static data paths
INDEX_PERSIST_DIR = "./index"
DOCS_DATA_DIR = "./data"
PROMPTS_DIR = "./prompts"

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = getenv("DATABRICKS_TOKEN")


# define LLM
llm = Databricks(
    model="databricks-meta-llama-3-70b-instruct",
    api_key=DATABRICKS_TOKEN,
    api_base="https://adb-2978037251816793.13.azuredatabricks.net/serving-endpoints",
    max_tokens=256,
)
# build or load index
if pathlib.Path(INDEX_PERSIST_DIR).exists():
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
    index = load_index_from_storage(storage_context, llm=llm)
else:
    documents = SimpleDirectoryReader(DOCS_DATA_DIR).load_data()
    index = TreeIndex.from_documents(documents, llm=llm)
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
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        system_prompt=system_prompt,
    )
    current_message = messages[-1]
    chat_history = messages[:-1]
    response = chat_engine.chat(str(current_message.content), chat_history=chat_history)
    return response


if __name__ == "__main__":
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="My patient_id is 1, Do I have diabetes?",
        )
    ]
    response = generate_response(messages=messages)
    print(response)
