import pathlib
from os import getenv

from dotenv import load_dotenv
from llama_index.core import (
    ChatPromptTemplate,
    KeywordTableIndex,
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.databricks import Databricks

load_dotenv()

INDEX_PERSIST_DIR = "./index"
DOCS_DATA_DIR = "./data"

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

# Define a custom prompt
# message_templates = [
#     ChatMessage(
#         content="You are an expert doctor with full experience assessing the specialism for a case given a set of symptoms given by a patience. You should always respond with only ONE WORD!",
#         role=MessageRole.SYSTEM,
#     ),
#     ChatMessage(
#         content="Redirect me to the right specialism based on the following symptoms: {symptoms}",
#         role=MessageRole.USER,
#     ),
# ]
# chat_template = ChatPromptTemplate.from_messages(message_templates)

# # you can create message prompts (for chat API)
# messages = chat_template.format_messages(llm=llm, symptoms="specialty")

# # # or easily convert to text prompt (for completion API)
# prompt = chat_template.format(llm=llm, symptoms="specialty")


chat_engine = index.as_chat_engine(
    llm=llm,
    memory=ChatMemoryBuffer.from_defaults(token_limit=10500),
    chat_mode=ChatMode.CONTEXT,
    system_prompt="You are an expert doctor with full experience assessing the specialism for a case given a set of symptoms given by a patience. You should always respond with only ONE specialist!",
)
response = chat_engine.chat(
    "I have a leg inflamed, what specialty should I go?", chat_history=[]
)
# chat_engine.reset()
# print(chat_engine.chat_history)
# response = chat_engine.chat("I have a leg inflamed, what specialty should I go?")
print(response)
