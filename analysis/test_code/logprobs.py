from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import math
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = "What is the capital of France?"

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    max_tokens=1000,
    temperature= 1,
    logprobs=True,
    top_logprobs= 1,
    messages=[
        {"role": "system", "content": "Respond to the user's question with a sentence."},
        {"role": "user", "content": prompt}
    ]
)

print(json.dumps(response.model_dump(), indent=1))

confidence = response.choices[0].logprobs.content[0].top_logprobs[0].logprob