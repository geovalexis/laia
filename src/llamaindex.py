from llama_index.core import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import pathlib
from llama_index.llms.databricks import Databricks
from os import getenv
from dotenv import load_dotenv

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


# get response from query
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("I have a leg inflamed, what specialty should I go?")
print(response)
