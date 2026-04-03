# in this we are going to translate the query into the language of the database we have some approach we gona try here

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pathlib import Path
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

user_query = input("Enter your question: ")

similar_variants =[]
SYSTEM_PROMPT = f"""
You are a helpful assistant that takes the user query {{user_query}} and then understands the context of the user query what it actually wants to ask and then after that it makes similar variants of user query which are semantically similar to the user query and then show to the user and that similar variants should be store in list and the list should be in json format as shown below and then return the list to the user and store that list to {similar_variants} list
 output format: {{
    "similar_variants":[ 
        "similar variant 1......",
        "similar variant 2......",
        "similar variant 3......",
        "similar variant 4......",
        "similar variant 5......"
    ]
}}

Rules:
1. The similar variants should be semantically similar to the user query.
2. The similar variants should be in json format as shown above.
3. The similar variants should be different from each other and should not be the same as the user query.
4. Don't add any unnecessary information in the output, only show the similar variants in json format as shown above.


Example:
user query: "What is the capital of France?"
output: 
{{
    "similar_variants":[ 
        "What is the capital city of France?",
        "Which city is the capital of France?",
        "Can you tell me the capital of France?",
        "What is the name of the capital of France?",
        "Where is the capital of France located?"
    ]
}}

"""

# create Embeddings through OpenAI API and specify the model and API key for authentication
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
)


# retriever = QdrantVectorStore.from_existing_collection(
#     url="http://localhost:6333",
#     collection_name="loyaltyos",
#     embedding=embedder,
# )

# relevant_chunks = retriever.similarity_search(query=user_query)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query},
]
response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account
        response_format={"type": "json_object"},
        messages=messages
    )
similar_variant = response.choices[0].message.content # storing the response 
similar_variants = json.loads(similar_variant)["similar_variants"] # parsing the response to get the similar variants list

for variant in similar_variants:
    print(variant)
