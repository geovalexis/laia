import pathlib
from os import getenv

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
    index = KeywordTableIndex.from_documents(documents, llm=llm)
    # Save the index for later use
    index.storage_context.persist(INDEX_PERSIST_DIR)

# Read system prompt
with open(pathlib.Path(PROMPTS_DIR) / "system.txt") as f:
    system_prompt = f.read()

# Create chat engine
chat_engine = index.as_chat_engine(
    llm=llm,
    memory=ChatMemoryBuffer.from_defaults(token_limit=10500),
    chat_mode=ChatMode.CONTEXT,
    system_prompt=system_prompt,
)
chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="I have a leg inflamed, what specialty should I go?",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Based on your symptoms, I would recommend that you see a Dermatologist.",
    ),
]
response = chat_engine.chat(
    "I have a leg inflamed, what specialty should I go?", chat_history=chat_history
)
print(response)
