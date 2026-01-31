from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI()

text ="Dogs are very loyal and cats are very caring"

# Create embedding
response = client.embeddings.create(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    input=text
)

print("Vector embeddings", response.data[0].embedding)