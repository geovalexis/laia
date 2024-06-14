from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = getenv("DATABRICKS_TOKEN")
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
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

print(chat_completion.choices[0].message.content)