from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()

# define the path to the PDF file
pdf_path = Path(__file__).parent / "loyaltyos.pdf"

# create a PyPDFLoader instance
loader = PyPDFLoader(file_path=pdf_path)

#stores the content as a document object
docs = loader.load()

# create a RecursiveCharacterTextSplitter instance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # specify the desired chunk size
    chunk_overlap=200,  # specify the desired chunk overlap
)

# split the documents into smaller chunks
split_docs = text_splitter.split_documents(documents=docs)

# print("Docs",len(docs))
# print("split_docs",len(split_docs))  # print the first 500 characters of the first chunk


# create Embeddings through OpenAI API and specify the model and API key for authentication
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=client.api_key,
)


# create a QdrantClient instance and inject the embedding model into the vector store

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="loyaltyos",
    embedding=embedder,
)

vector_store.add_documents(documents=split_docs)

print("Documents added to the vector store successfully.")

# retriever = QdrantVectorStore.from_existing_collection(
#     url="http://localhost:6333",
#     collection_name="loyaltyos",
#     embedding=embedder,
# )
question = input("Enter your question: ")
#extracting the relevant chunks from the vector store
# relevant_chunks = retriever.similarity_search(query=question)

#storing the content of the relevant chunks in a list
# contents = [doc.page_content for doc in relevant_chunks]

#writing the system prompt for the RAG model
# SYSTEM_PROMPT = f"""
# You are a helpful assistant who is expert in answering the question based on the given context.

# #added the content of the relevant chunks in the system prompt
# content: {contents}


# Rules:
# 1. Always answer the question based on the given content.
# 2. If you don't know the answer based on the given content then say "I don't know" and don't try to answer the question.
# 3. Always try to answer the question in a concise manner.

# Example:
# Question: What is loyaltyos?
# Answer: LoyaltyOS is a customer loyalty management platform that helps businesses create and manage customer loyalty programs. It provides features such as customer segmentation, personalized rewards, and analytics to help businesses increase customer retention and engagement.

# Question: What is the capital of France?
# Answer: I don't know.

# Question: What is the weather of new york?
# Answer: I don't know.

# question: What is the main feature of loyaltyos?
# Answer: The main feature of LoyaltyOS is its ability to create and manage customer loyalty programs.

# """
# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": question},
# ]
# response = client.chat.completions.create(
#         model="gpt-4o-mini",  # or another JSON-capable model in your account
#         # response_format={"type": "json_object"},
#         messages=messages
#     )
# print(response.choices[0].message.content)