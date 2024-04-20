from openai import OpenAI
import os

CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def llm(prompt, engine, logit_bias = {}, temperature = 0.0, images=None, stop=["\n"]):
    
    if engine == 'gpt-4':
        response = CLIENT.chat.completions.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can plan household tasks."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            logit_bias=logit_bias
        )
        return response.choices[0].message.content
    
    elif engine == 'gpt-4v':
        response = CLIENT.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=100,
            temperature=temperature,
            logit_bias=logit_bias
        )
        return response.choices[0].message.content

    else:
        response = CLIENT.completions.create(
            model="gpt-3.5-turbo-instruct", #"text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            logit_bias=logit_bias
        )
        return response.choices[0].text