#In this code we will be using llm to divide the user query into some similar variants and then we will go one by one and retrieve the respective chunks from the qdrant db and then we will use that retrieved information with the next user query to get the better chunks and more accuracy and we will keep on doing this until we reach to that final sub query and then we will use all the retrieved chunks along with the actual user query and give them to the llm to get the more accurate answer to the user query.


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

#storing the similar variants in a list 
similar_variants = [ ]

splitter_prompt = """
You are a helpful assistant that prepares queries that will be sent to a search component.
Sometimes, these queries are very complex.
Your job is to simplify complex queries into multiple queries that can be answered
in isolation to eachother.

If the query is simple, then keep it as it is.
Examples
1. Query: Did Microsoft or Google make more money last year?
   Decomposed Questions: [Question(question='How much profit did Microsoft make last year?', answer=None), Question(question='How much profit did Google make last year?', answer=None)]
2. Query: What is the capital of France?
   Decomposed Questions: [Question(question='What is the capital of France?', answer=None)]
3. Query: {{question}}
   Decomposed Questions:

Examples:
"If my future wife has the same first name as the 15th first lady of the United States' mother
and her surname is the same as the second assassinated president's mother's maiden name,
what is my future wife's name?"

This query requires multiple interconnected steps:
1. Identify the 15th first lady of the United States
2. Find the 15th first lady's mother's first name
3. Identify the second assassinated president
4. Find that president's mother's maiden name
5. Combine the two names to form the final answer

RULE:
- Decompose the query into multiple simpler questions that can be answered independently.
- If the query is already simple, return it as a single question.
- Format the output as a list of Question objects, where each Question has a 'question' field and an 'answer' field (which should be None in the output).
- Ensure that the questions are clear and concise, and that they can be answered without needing to reference each other.
- Do not include any reasoning steps or explanations in the output, only the final list of questions.
- Always return the output in the JSON-specified format, even if the query is simple. For example, if the query is "What is the capital of France?", the output should be in json format like: [Question(question='What is the capital of France?', answer=None)]
"""

# storing the system prompt and user query in list 
messages = [
    {"role": "system", "content": splitter_prompt},
    {"role": "user", "content": user_query},
]

response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account 
        # response_format={"type": "json_object"},
        messages=messages  # type: ignore[arg-type]
    )
similar_variant = response.choices[0].message.content # storing the response 

# print(similar_variant) # printing the response
# =====================================================
# PARSE SUBQUERIES FROM SIMILAR_VARIANT
# =====================================================
# Extract subqueries from the response (parsing JSON or structured format)

# Try to parse as JSON first, if fails, use regex to extract questions
try:
    parsed_questions = json.loads(similar_variant)
    subqueries = [q["question"] if isinstance(q, dict) else str(q) for q in parsed_questions]
except:
    # Use regex to extract questions between quotes or Question(question='...')
    subqueries = re.findall(r"[Qq]uestion\([Qq]uestion='([^']+)'", similar_variant)
    if not subqueries:
        subqueries = re.findall(r"'([^']+)'", similar_variant)
    if not subqueries:
        subqueries = [similar_variant]  # Fallback to original if parsing fails

print(f"\n{'='*60}")
print(f"Extracted Subqueries: {len(subqueries)}")
for i, sq in enumerate(subqueries, 1):
    print(f"  {i}. {sq}")
print(f"{'='*60}\n")

# =====================================================
# SETUP QDRANT CONNECTION AND EMBEDDINGS
# =====================================================
# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=api_key,
)

# Reuse one vector store instance for all subquery iterations
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="loyaltyos",
    url="http://localhost:6333"
)

# =====================================================
# ITERATIVE SUBQUERY PROCESSING
# =====================================================
all_retrieved_contexts = []  # Store all contexts for final answer generation
previous_output = ""  # Store output from previous iteration

for idx, subquery in enumerate(subqueries, 1):
    print(f"\n{'='*60}")
    print(f"Processing Subquery {idx}/{len(subqueries)}")
    print(f"{'='*60}")
    
    # If not the first query, combine with previous output
    if idx > 1:
        combined_input = f"""
Previous Query Output: {previous_output}

Current Subquery: {subquery}
"""
        print(f"Combined Input (with previous output):")
        print(combined_input[:200] + "..." if len(combined_input) > 200 else combined_input)
    else:
        combined_input = subquery
        print(f"Current Subquery: {subquery}")
    
    # =====================================================
    # CREATE EMBEDDINGS FOR CURRENT INPUT
    # =====================================================
    try:
        # Generate embedding for the combined input (or just subquery if first iteration)
        query_embedding = embeddings.embed_query(combined_input)
        print(f"✓ Generated embedding (dimension: {len(query_embedding)})")
        
        # =====================================================
        # PERFORM SIMILARITY SEARCH ON QDRANT
        # =====================================================
        results = vectorstore.similarity_search(combined_input, k=5)
        
        print(f"✓ Similarity search completed (found {len(results)} similar contexts)")
        
        # =====================================================
        # CHECK IF RESULTS ARE EMPTY (OUT OF KNOWLEDGE BASE)
        # =====================================================
        out_of_scope = False
        if not results or len(results) == 0:
            out_of_scope = True
            print(f"⚠ WARNING: No similar documents found in Qdrant for this subquery!")
            print(f"⚠ This question appears to be OUT OF SCOPE for the knowledge base")
        
        # =====================================================
        # STORE RETRIEVED CONTEXT FOR THIS SUBQUERY
        # =====================================================
        context_for_subquery = {
            "subquery": subquery,
            "iteration": idx,
            "retrieved_results": results,
            "previous_output": previous_output if idx > 1 else None,
            "out_of_scope": out_of_scope
        }
        all_retrieved_contexts.append(context_for_subquery)
        
        # =====================================================
        # GENERATE OUTPUT FOR THIS SUBQUERY
        # =====================================================
        # Use LLM to generate answer based on retrieved context
        if out_of_scope:
            # If no relevant results found, inform the user
            context_text = "NO RELEVANT DOCUMENTS FOUND IN KNOWLEDGE BASE"
            llm_messages = [
                {"role": "system", "content": "You are a helpful assistant. If no context is provided or the context says 'NO RELEVANT DOCUMENTS FOUND', then tell the user that this question is beyond the scope of your knowledge base and you don't have information about it."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {subquery}\n\nPlease inform the user if this is out of scope:"}
            ]
        else:
            context_text = "\n".join([str(r) for r in results[:3]])  # Top 3 results
            llm_messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {subquery}\n\nAnswer:"}
            ]
        
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=llm_messages
        )
        
        previous_output = llm_response.choices[0].message.content
        print(f"✓ Generated response for subquery {idx}")
        print(f"Response preview: {previous_output[:150]}..." if len(previous_output) > 150 else f"Response: {previous_output}")
        
    except Exception as e:
        print(f"✗ Error processing subquery {idx}: {str(e)}")
        previous_output = ""

print(f"\n{'='*60}")
print("COMPLETED: All subqueries processed")
print(f"{'='*60}\n")

# =====================================================
# FINAL ANSWER GENERATION
# =====================================================
print("\nGenerating final comprehensive answer...")

# Check if any subqueries were out of scope
out_of_scope_queries = [ctx for ctx in all_retrieved_contexts if ctx.get("out_of_scope", False)]
any_out_of_scope = len(out_of_scope_queries) > 0

# Prepare all retrieved contexts for final LLM
final_contexts_summary = "\n\n".join([
    f"Subquery {ctx['iteration']}: {ctx['subquery']}\n" + 
    (f"⚠ [OUT OF SCOPE - No similar documents found in knowledge base]" if ctx.get("out_of_scope") else f"Context: {str(ctx['retrieved_results'])}")
    for ctx in all_retrieved_contexts
])

# Build final message with warning if needed
scope_warning = ""
if any_out_of_scope:
    scope_warning = f"\n\n⚠ IMPORTANT: {len(out_of_scope_queries)} out of {len(all_retrieved_contexts)} subqueries have questions that are OUT OF SCOPE (not found in the knowledge base). Please inform the user about these limitations."

final_messages = [
    {"role": "system", "content": "You are a helpful assistant. Provide a comprehensive answer based on all the provided information. If any subqueries are marked as OUT OF SCOPE, clearly inform the user that those parts of their question cannot be answered from the available knowledge base."},
    {"role": "user", "content": f"Original Question: {user_query}\n\nRetrieved Information and Subquery Results:\n{final_contexts_summary}{scope_warning}\n\nProvide a comprehensive final answer. If parts of the question are out of scope, clearly mention which parts cannot be answered:"}
]

final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=final_messages
)

final_answer = final_response.choices[0].message.content
print(f"\n{'='*60}")
print("FINAL ANSWER:")
print(f"{'='*60}")
print(final_answer)

# =====================================================
# KNOWLEDGE BASE COVERAGE SUMMARY
# =====================================================
if any_out_of_scope:
    print(f"\n{'='*60}")
    print("⚠ KNOWLEDGE BASE COVERAGE SUMMARY:")
    print(f"{'='*60}")
    print(f"Total Subqueries: {len(all_retrieved_contexts)}")
    print(f"In Scope (Found): {len(all_retrieved_contexts) - len(out_of_scope_queries)}")
    print(f"Out of Scope (Not Found): {len(out_of_scope_queries)}")
    print(f"\nOut of Scope Questions:")
    for ctx in out_of_scope_queries:
        print(f"  • {ctx['subquery']}")
    print(f"{'='*60}")



