import openai

client = openai.OpenAI(
    base_url="https://llmproxy.uva.nl",
)

response = client.chat.completions.create(
    model="Qwen-2.5-VL", # model to send to the proxy, see below for options available
    messages = [{
        "role": "user",
        "content": "this is a test request, write a short poem"
    }]
)

print(response.choices[0].message.content)