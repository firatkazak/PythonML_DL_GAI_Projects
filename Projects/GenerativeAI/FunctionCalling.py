import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def get_current_weather(location, unit="fahrenheit"):
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    if "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    {"role": "user",
     "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}
]

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
print("Response: ", response_message)

messages.append(response_message)
tool_calls = response_message.tool_calls
print("Tool Call: ", tool_calls)

available_functions = {"get_current_weather": get_current_weather}

for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit")
    )
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response
        }
    )

print("Mesaj: ", messages)

second_response = client.chat.completions.create(model="gpt-3.5-turbo-1106",
                                                 messages=messages,
                                                 )
result = second_response.choices[0].message.content
print("Çıktı: ", result)
