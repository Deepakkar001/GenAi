import json ,requests, os ,subprocess
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()


def get_weather(city: str):
    print("Tool Called: get_weather",city)
    response = requests.get(f"https://api.weatherapi.com/v1/current.json?key=4e74b0518062432c974140447251409&q={city}")
    if response.status_code != 200:
        return f"Failed to get weather data for {city}. Please try again later."
    data = response.json()
    temp_c = data["current"]["temp_c"]
    return f"The Weather of {city} is {temp_c} degree celsius"


def run_command(command: str):
   result = os.system(command=command)
   return f"Command executed with exit code {result}"


Available_tools ={
    "get_weather" :{
        "fn": get_weather,
        "description": "This is function which takes city name as input and return the current weather of that city"
    },
    "run_command": {
        "fn": run_command,
        "description": "This is function which takes a command as input and then detect which os it is like it is mac os or windows os or ubuntu os and then accordingly think which command will suitable for that os  and then  executes it if that current command is not getting executed then it wil try with another command and then return the output of the command"
    }

}

system_prompt= f""" 
  You are an helpful AI assistant who is expert in resolving user query.
  You work on start,plan,action,observe modes.
  For the given user query and available tools,plan the step by step execution ,based on the planning select the relevant tool from the available tool.And based on the tool selection you perform an action to call the tool and then wait for the observation and based on the observation from the tool call the user query and then finally give the result to the user.

  Rules:
    1. Follow the strict JSON output as per Output Schema.
    2. Always perform one step at a time and wait for next input
    3. Carefully analyse the user query and then plan the execution based on the available tools.

  Output JSON Format:
  {{
     "step":"string",
        "content":"string",
        "function":"The name of function if the step is action else null",
        "input":"The input parameter for the function if the step is action else null",
  }}

  Available Tools:
  - get_weather : This is function which takes city name as input and return the current weather of that city

  - run_command: This is function which takes a command as input and then detect which os it is like it is mac os or windows os or ubuntu os and then accordingly think which command will suitable for that os  and then  executes it if that current command is not getting executed then it wil try with another command and then return the output of the command

  Example:
  User Query: What is the weather of new york?
  Output:{{"step":"plan","content":"The user is interested in weather information of new york"}}
  Output:{{"step":"plan","content":"From the available tools i should call get_weather"}}
  Output:{{"step":"action","function":"get_weather","input":"new york"}}
  Output:{{"step":"observe","output":"The weather of new york is 30 degree celsius and sunny"}}
  Output:{{"step":"result","content":"The weather of new york is 30 degree celsius and sunny"}}

"""

messages=[{"role" : "system","content":system_prompt},]
while True:
    user_query = input("Enter your query: ")
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
        model="gpt-4o-mini",  # or another JSON-capable model in your account
        response_format={"type": "json_object"},
        messages=messages
    )
        
        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        if parsed_output["step"] == "plan":
            print(f"Planning 💭 : {parsed_output.get('content')}")
            continue
        if parsed_output["step"] == "action":
            tool_function = parsed_output.get("function")
            tool_input = parsed_output.get("input")   
            if Available_tools.get(tool_function ,False) != False:
                output= Available_tools[tool_function].get("fn")(tool_input)
                messages.append({"role": "assistant", "content": json.dumps({"step":"observe","output":output})})
                continue
        if parsed_output["step"] == "result":
            print(f"Final Result 🎯 : {parsed_output.get('content')}")
            break
        
    
