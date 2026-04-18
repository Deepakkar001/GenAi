# In this we will take the user query and we will ask llm to answer from his side and make a hypothectical document which is relevant to the user query and then we will make the embeddings of that hypothectical document and then we will perform the similarity search on the vector database and then we will extract the relevant chunks from the vector database and then we will use that context and the user query to generate the final answer.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
import os
import json


from dotenv import load_dotenv
from openai import OpenAI

# load the environment variables from the .env file and get the OpenAI API key  
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI()

# taking the user query as input from the user
user_query = input("Enter your question: ")

Hyde_prompt = f"""
You are a helpful assistant that takes the user query {{user_query}} and then understands the context of the user query what it actually wants to ask and then after that it makes a hypothectical document which is relevant to the user query and then return that hypothectical document to the user in json format as shown below and then store that hypothectical document in a variable called hypothectical_document

output format: {{
    "hypothetical_document": "hypothetical document here"
}}

Rules:
1. The hypothectical document should be relevant to the user query.
2. The hypothectical document should be in json format as shown above.
3. The hypothectical document should be comprehensive and should cover the general topic of the user query.
4. The hypothectical document should not contain any specific details from the user query.
5. The hypothectical document should not be contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
6. The hypothectical document should be in the form of a document and should not be a question or a statement.
7. The output should be in json format as shown above.
Example:
user query: "What is the capital of France?"
output: 
{{
    "hypothetical_document": "The capital of France is Paris. Paris is the largest city in France and is known for its rich history, culture, and architecture. It is located in the northern part of the country and is a major center for art, fashion, and gastronomy. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also a popular tourist destination, attracting millions of visitors each year."
}}
"""

messages = [
    {"role": "system", "content": Hyde_prompt},
    {"role": "user", "content": user_query},
]

response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account 
        response_format={"type": "json_object"},
        messages=messages  # type: ignore[arg-type]
    )
hypothetical_document = response.choices[0].message.content # storing the response 

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
)

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="loyaltyos",
    embedding=embedder,
)

# 1) Parse LLM JSON output
abstract_obj = json.loads(hypothetical_document) if isinstance(hypothetical_document, str) else hypothetical_document
hypothetical_document = (
    abstract_obj.get("hypothetical_document")
    or abstract_obj.get("hypothectical_document")
    or ""
).strip()

if not hypothetical_document:
    raise ValueError("hypothetical_document is empty in model output")

print("Hypothetical document:", hypothetical_document)

# 2) Create embedding for hypothetical document
hypothetical_document_vector = embedder.embed_query(hypothetical_document)
print("Embedding dimension:", len(hypothetical_document_vector))

# 3) Similarity search using the vector
results = retriever.similarity_search_by_vector(hypothetical_document_vector, k=5)


print("Retrieved docs:", len(results))
for i, doc in enumerate(results, 1):
    print(f"\nDoc {i}:")
    print(doc.page_content[:400])

# 4) Build context from retrieved chunks
context_blocks = []
for i, doc in enumerate(results, 1):
    text = doc.page_content.strip() if doc.page_content else ""
    if text:
        context_blocks.append(f"Chunk {i}:\n{text}")

context_text = "\n\n".join(context_blocks)

# 5) Final answer generation using context + original user query
if not context_text:
    print("\nNo relevant context found in vector database. Cannot generate grounded final answer.")
else:
    final_system_prompt = """
You are a helpful and accurate assistant.
Use only the provided context chunks {{context_text}} to answer the user question {{user_query}} and then you read the context and then you give the answer to the user query {{user_query}} based on the context and then also suggest some relevant answer which are related to the user query and also related to the context and then give them in a json format as shown below and then return that json to the user.

Rules:
1. Always answer the question based on the given context.
2. If you don't know the answer based on the given context then say "I don't know" and don't try to answer the question.
3. Always try to answer the question in a concise manner.
4. The answer should be in json format as shown below.
output format:
{{
    "answer": "answer to the user query based on the context",
    "suggestions": [
        "relevant suggestion 1",
        "relevant suggestion 2",
        "relevant suggestion 3"
    ]
}}
"""
final_messages = [
        {"role": "system", "content": final_system_prompt},
        {
            "role": "user",
            "content": (
                f"User Question:\n{user_query}\n\n"
                f"Hypothetical Document Used for Retrieval:\n{hypothetical_document}\n\n"
                f"Retrieved Context Chunks:\n{context_text}\n\n"
                "Now provide the best possible grounded answer to the user question."
            ),
        },
    ]

final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=final_messages,
    )
final_answer = final_response.choices[0].message.content
print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(final_answer)
