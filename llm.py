from openai import OpenAI
from openai import AzureOpenAI
import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

class LLM:
    def __init__(self, model):
        self.model = model
    
    def generate_response(self, user_input, history=[], prompt="一律使用英文回答"):
        messages = [{"role": "system", "content": prompt}]
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})
        
        if any(keyword in self.model for keyword in ['ffm', 'taide', 'deepseek', 'gte', 'qwen', 'meta', 'bge', 'jina']):
            API_KEY = os.getenv('TWSC_API_KEY')
            BASE_URL = "https://api-ams.twcc.ai/api/models"
            
            client = OpenAI(
                api_key = API_KEY,
                base_url = BASE_URL
            )
            
            response = client.chat.completions.create(
                model = self.model,
                temperature = 0.2,
                max_tokens = 4096,
                top_p = 0.95,
                messages = messages,
                stream = False
            )
            
            return response.choices[0].message.content
        
        if any(keyword in self.model for keyword in ['gpt']):
            client = AzureOpenAI(
                azure_endpoint=os.getenv('AZURE_ENDPOINT'),
                api_key=os.getenv('AZURE_API_KEY'),
                api_version=os.getenv('AZURE_API_VERSION'),
            )
            
            response = client.chat.completions.create(
                model = self.model,
                temperature = 0.2,
                top_p = 0.95,
                messages = messages,
                stream = False
            )
            
            return response.choices[0].message.content
        
        else:
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }

            try:
                response = requests.post(
                    f'{os.getenv('OLLAMA_BASE_URL')}/api/chat',
                    headers=headers,
                    data=json.dumps(data),
                )
                response.raise_for_status()
                response_data = response.json()

                return response_data.get('message').get('content').strip()
            
            except Exception as e:
                print(f"Error generating response: {e}")
                return None
            
        
if __name__ == "__main__":
    model = "gpt-4o-mini"
    llm = LLM(model)
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "Hi"}]
    user_input = "你是誰"
    response = llm.generate_response(user_input, history)
    print(response)
        