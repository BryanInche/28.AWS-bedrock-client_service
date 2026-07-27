from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hola, ¿qué modelo eres?"}]
)
print(response.choices[0].message.content)

for model in client.models.list().data:
    print(model.id)