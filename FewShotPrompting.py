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

result = client.chat.completions.create(
    model = "gpt-5",
    messages =[
        {"role" : "system","content":system_prompt},
        {"role":"user","content":"what is 2+2*0"},
        {"role":"assistant","content":json.dumps({"step": "Analyse","content": "The user asks to evaluate the arithmetic expression '2+2*0'. This requires applying the order of operations (multiplication before addition)."})},
        {"role":"assistant","content":json.dumps({"step":"Think","content":"Plan to solve '2+2*0' using order of operations: 1) Parse the expression and note there is addition and multiplication. 2) Apply multiplication before addition (BODMAS/PEMDAS). 3) Compute the multiplication: 2*0 = 0. 4) Substitute back into the expression: 2 + 0. 5) Perform the remaining addition: 2 + 0 = 2. 6) Confirm there are no further operations and prepare this value for output."})},
        {"role":"assistant","content":json.dumps({"step":"Result","content":"2"})},
        {"role":"assistant","content":json.dumps({"step": "Output", "content": "2"})},
    ]
)

print(result.choices[0].message.content)
