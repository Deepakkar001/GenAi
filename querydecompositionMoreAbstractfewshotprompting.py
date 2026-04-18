#In this code first we will take the user query and then we will use an llm to make it more abstract then according to that make the embeddings and then perform the similarity search and then we will extract the relevant chunks and then we will use that context and the user query to generate the final answer.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
import os
import json
import re

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

Abstract_query_prompt = """
You are an expert of world knowledge. I am going to ask you a question{user_query}.
Your response should be comprehensive and not contradicted with the
following context if they are relevant. Otherwise, ignore them if they are
not relevant.

Examples:
"input": "Could the members of The Police perform lawful arrests?"
"output": "What can the members of The Police do?"

"input": "Jan Šindel was born in what country?"
"output": "What is Jan Šindel’s personal history?"

"input": "What are the side effects of drug X in elderly patients?"
"output": "What are the general side effects of drug X?" or "How does drug X affect the human body?"

Now rewrite the next input accordingly. Return only the rewritten question.
RULES:
1. The rewritten question should be more abstract than the original question.
2. The rewritten question should not contain any specific details from the original question.
3. The rewritten question should be comprehensive and should cover the general topic of the original question.
4. The rewritten question should not be contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
5. The rewritten question should be in the form of a question and should not be a statement.
6. The output should be in json format as shown below.
OUTPUT FORMAT:
{{
    "abstract_query": "rewritten question here"
}}
"""

# storing the system prompt and user query in list 
messages = [
    {"role": "system", "content": Abstract_query_prompt},
    {"role": "user", "content": user_query},
]

response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account 
        response_format={"type": "json_object"},
        messages=messages  # type: ignore[arg-type]
    )
Abstract_variant = response.choices[0].message.content # storing the response in a variable

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
abstract_obj = json.loads(Abstract_variant) if isinstance(Abstract_variant, str) else Abstract_variant
abstract_query = abstract_obj.get("abstract_query", "").strip()

if not abstract_query:
    raise ValueError("abstract_query is empty in model output")

print("Abstract query:", abstract_query)

# 2) Create embedding for abstract query
abstract_vector = embedder.embed_query(abstract_query)
print("Embedding dimension:", len(abstract_vector))

# 3) Similarity search using the vector
results = retriever.similarity_search_by_vector(abstract_vector, k=5)

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
Use only the provided context chunks to answer the user question.
If the context is insufficient for any part, clearly say what is missing.
Keep the answer clear, concise, and directly focused on the user question.
"""

    final_messages = [
        {"role": "system", "content": final_system_prompt},
        {
            "role": "user",
            "content": (
                f"User Question:\n{user_query}\n\n"
                f"Abstract Query Used for Retrieval:\n{abstract_query}\n\n"
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