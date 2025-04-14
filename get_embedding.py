import openai
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


# def get_embedding(text):
#     return openai.Embedding.create(
#         input=[text.replace("\n", " ")], model="text-embedding-ada-002"
#     )["data"][0]["embedding"]


def get_embedding(text):
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=[text.replace("\n", " ")], model="text-embedding-ada-002"
    )
    return response.data[0].embedding


# empbedding = get_embedding("cat")
# pprint(empbedding)
# print(len(empbedding))
