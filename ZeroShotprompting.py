from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


//ZERO SHOT prompting 
completion = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "developer", "content": "Talk like a senior dev."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print(completion.choices[0].message.content)