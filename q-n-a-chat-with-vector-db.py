import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv
from ChatTemplate import ChatTemplate
load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(
    "pod-racing",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
    ),
)

for f in os.listdir("./racing"):
    path = os.path.join("./racing", f)
    with open(path, "r") as f:
        text = f.read()

    text = text.replace("\n", " ")

    collection.add(documents=[text], ids=[path])

collection = client.get_collection(
    'pod-racing',
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name="text-embedding-ada-002"
    ))

chat = ChatTemplate(
    {'messages': [{'role': 'system', 'content': 'You are a Q&A AI.'},
                  {'role': 'system', 'content': 'Here are some facts that can help you answer the following question: {{data}}'},
                  {'role': 'user', 'content': '{{prompt}}'}]})
while True:
    prompt = input('user: ')
    if prompt == 'exit':
        break

    data = collection.query(query_texts=[prompt], n_results=1)[
        'documents'][0][0]

    message = chat.completion(
        {'data': data, 'prompt': prompt}).choices[0].message

    print(f'{message.role}: {message.content}')
