from groq import Groq 
client = Groq(api_key='gsk_uFJlnp11MvYLJrq7NCVQWGdyb3FY2az46zSO7jiTvmmB2iLfVgD') 
response = client.chat.completions.create( 
    model='llama3-8b-8192', 
    messages=[{'role': 'user', 'content': 'Hello'}] 
) 
print(response.choices[0].message.content) 
