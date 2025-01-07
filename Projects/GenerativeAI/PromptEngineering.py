from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()


def chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(model=model,
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }],
                                              temperature=temperature,
                                              )
    print(response.choices[0].message.content)


chat_completion("Give me a suggenstion for the main course for today's lunch.")
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian,\
 and I don't like green vegetable. I am conscientious\
 about eating healthily. Task: Give me a suggenstion for the main\
 course for today's lunch.         
"""
chat_completion(prompt)

prompt = """
Context: I do 2 hours of sport a day. I am vegetarian,\
 and I don't like green vegetable. I am conscientious\
 about eating healthily. Task: Give me a suggenstion for the main\
 course for today's lunch.\
 Don't perform the requested task! Instead, can you ask me queations\
 about the context so that when I answer, you can perform the\
 requested task more efficiently?               
"""
chat_completion(prompt)

prompt = """
Context: I do 2 hours of sport a day. I am vegetarian,\
 and I don't like green vegetable. I am conscientious\
 about eating healthily. Task: Give me a suggenstion for the main\
 course for today's lunch.\
 With this suggestion, I also want a table with two columns\
 where each row contains an ingredient from the main course.\
 The first column in the table is the name of the ingredient.\
 The second column of the table is the number of grams of that\
 ingredient needed for one person. Do not give the recipe for\
 preparing the main course.         
"""
chat_completion(prompt)

prompt = """
 Role: You are a nutritionist designing healthy diets for\
 high-performance athletes. You take into account the nutrition\
 needed for a good recovery.\
 Context: I do 2 hours of sport a day. I am vegetarian,\
 and I don't like green vegetable. I am conscientious\
 about eating healthily.\ 
 Task: Give me a suggenstion for the main course for today's lunch.\
 With this suggestion, I also want a table with two columns\
 where each row contains an ingredient from the main course.\
 The first column in the table is the name of the ingredient.\
 The second column of the table is the number of grams of that\
 ingredient needed for one person. Do not give the recipe for\
 preparing the main course.         
"""
chat_completion(prompt)
