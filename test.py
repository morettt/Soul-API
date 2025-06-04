from openai import OpenAI

API_KEY = 'sk-5xSeGGKp6ZqBFip8wqkjiXBdA0sQYsL9eUDJs3kOhEosB8Tk'
API_URL = 'http://morellm.natapp1.cc/v1'
messages=[{'role':'system','content':'模仿傲娇和我对话'}]

def chat():
    while True:
        user = input('你：')
        messages.append({'role': 'user', 'content': user})

        client = OpenAI(api_key=API_KEY, base_url=API_URL)

        response = client.chat.completions.create(
            messages=messages,
            model='gemini-2.0-flash',
            stream=True
        )

        full_assistant = ''
        print('AI：',end='')
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                ai_response = chunk.choices[0].delta.content
                print(ai_response,end='')
                full_assistant += ai_response

        print()

        messages.append({'role': 'assistant', 'content': full_assistant})

if __name__ == '__main__':
    chat()