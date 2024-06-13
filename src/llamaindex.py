from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from llama_index.llms.databricks import Databricks
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = getenv("DATABRICKS_TOKEN")

documents = SimpleDirectoryReader("data").load_data()

# define LLM
llm = Databricks(
    model="databricks-dbrx-instruct",
    api_key=DATABRICKS_TOKEN,
    api_base="https://adb-2978037251816793.13.azuredatabricks.net/serving-endpoints",
    max_tokens=256,
)
# build index
index = KeywordTableIndex.from_documents(documents, llm=llm)

# get response from query
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Hola, quien soy?")
print(response)
