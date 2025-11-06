from dotenv import load_dotenv
import os

import httpx
import openai
from openai import OpenAI

print("httpx version:", httpx.__version__)
print("openai version:", openai.__version__)

load_dotenv()
print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from Srini's test"}],
)

print(resp.choices[0].message.content)
