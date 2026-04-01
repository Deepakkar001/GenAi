import json

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt= """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

for the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

Follow the steps in sequence that is "Analyse","Think","Output","Validate" and finally "Result".

Rules:
1.Follow the strict JSON output as per Output Schema.
2.Always perform one step at a time and wait for next input
3.Carefully analyse the user query

Output Format:
{{step:"string",content:"string"}}

Example:
Input: what is 2+2.
Output: {{step: "Analyse",content: "Alright! The user is intersted in maths query and he is talking a basic mathmetical question}}

Output: {{step: "Think",content: "To perform the mathmetical operation i must go from left to right and should obey the BODMAS rule we have in our math and do the calculation"}}

Output: {{step: "Output",content: "{result calculated by me in the previous steps}"}}

Output: {{step: "Validate",content: " Cross checking the answer and verifying with my Knowledge base"}}

Output: {{step:"Result", content:"This is the {verfied answer}answer}}

 """

messages=[
    {"role" : "system","content":system_prompt},]

query = input("Enter your query: ")
messages.append({"role":"user","content":query})

while True:
    response = client.chat.completions.create(
    model="gpt-4o-mini",  # or another JSON-capable model in your account
    messages=messages,
    response_format={"type": "json_object"}
) 

    try:
        parsed_response = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        break
    
    messages.append({"role":"assistant","content":json.dumps(parsed_response)})

    if parsed_response.get("step") == "Result":
        print(f"Final Result: {parsed_response.get('content')}")
        query = input("Enter your query: ")
        messages.append({"role":"user","content":query})

    print(f"{parsed_response.get('step')} : {parsed_response.get('content')}")
