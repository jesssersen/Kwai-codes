
import os
import base64
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://mxxxxx")
deployment = os.getenv("DEPLOYMENT_NAME", "mmu-axxxxx")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview", # 可参考官网https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
)

# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "developer",
        "content": [
            {
                "type": "text",
                "text": "你是一个帮助用户查找信息的 AI 助手。"
            }
        ]
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

# Generate the completion
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_completion_tokens=16384,
    stop=None,
    stream=False
)

print(completion.to_json())
    