# in this we are going to rank the similar variants of questions output and then we will give that to the llm model and then according to that we will give the answer to the user query based on the relevant chunks from the vector store 
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

# load the environment variables from the .env file and get the OpenAI API key  
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI()

# taking the user query as input from the user
user_query = input("Enter your question: ")

#storing the similar variants in a list 
similar_variants =[]

# define the system prompt for the llm model to generate relevant questions
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
)

# storing the system prompt and user query in list 
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query},
]

#connecting to the llm model and getting the response
response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account 
        response_format={"type": "json_object"},
        messages=messages  # type: ignore[arg-type]
    )
similar_variant = response.choices[0].message.content # storing the response 
if similar_variant is None:
    raise RuntimeError("LLM returned an empty response")
similar_variants = json.loads(similar_variant)["similar_variants"] # parsing the response to get the similar variants list

#printing the similar variants
for variant in similar_variants:
    print(variant)

#retrieviing the relevant chunks from qdrant db using vector of similar variants
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="loyaltyos",
    embedding=embedder,
)

# reciprocal-rank fusion by chunk text
fused_chunks = {}
rrf_k = 60  # common RRF constant

for variant in similar_variants:
    docs = retriever.similarity_search(variant, k=3)  # text query is fine
    for rank_pos, doc in enumerate(docs, start=1):
        rrf_score = 1.0 / (rrf_k + rank_pos)
        chunk_text = doc.page_content.strip()
        chunk_key = " ".join(chunk_text.lower().split())

        if chunk_key not in fused_chunks:
            fused_chunks[chunk_key] = {
                "chunk": chunk_text,
                "rrf_score": 0.0,
                "best_rank": rank_pos,
                "sources": [],
            }

        fused_chunks[chunk_key]["rrf_score"] += rrf_score
        fused_chunks[chunk_key]["best_rank"] = min(fused_chunks[chunk_key]["best_rank"], rank_pos)
        fused_chunks[chunk_key]["sources"].append({
            "variant": variant,
            "rank_in_variant": rank_pos,
        })

# global ranking across all variant results, fused per chunk
ranked_context = sorted(
    fused_chunks.values(),
    key=lambda item: (item["rrf_score"], -item["best_rank"]),
    reverse=True,
)

# keep top context for final RAG prompt
top_n = 5
ranked_context = ranked_context[:top_n]

print(json.dumps(ranked_context, indent=2))

SYSTEM_PROMPT_RAG = f"""
You are a grounded question-answering assistant.

User query:
{user_query}

Ranked evidence from retrieval fusion, ordered from strongest to weakest:
{json.dumps(ranked_context, indent=2)}

How to use the evidence:
1. Prefer higher-ranked items first.
2. Use repeated support across sources as stronger evidence.
3. If multiple chunks conflict, trust the higher-ranked or more repeated evidence.
4. Ignore irrelevant or low-signal text.
5. Answer only from the ranked evidence; do not use outside knowledge.
6. If the evidence is insufficient, return "I don't know".

Output rules:
1. Be concise and direct.
2. Return strict JSON only.
3. Use this schema:
{{
    "answer": "...",
    "support": [
        {{"rank": 1, "variant": "..."}},
        {{"rank": 2, "variant": "..."}}
    ]
}}
4. Keep `support` short and include only the ranks you actually used.
"""
messages_rag = [
    {"role": "system", "content": SYSTEM_PROMPT_RAG},
    {"role": "user", "content": user_query},
]
response_rag = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account 
        response_format={"type": "json_object"},
    messages=messages_rag  # type: ignore[arg-type]
    )   
final_answer = response_rag.choices[0].message.content
print(final_answer)







