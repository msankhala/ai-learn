import json
from get_embedding import get_embedding
from ChatTemplate import ChatTemplate

embeddings = json.load(open("embeddings.json", "r"))


def cosine_distance(a, b):
    return 1 - sum([a_i * b_i for a_i, b_i in zip(a, b)]) / (
        sum([a_i**2 for a_i in a]) ** 0.5 * sum([b_i**2 for b_i in b]) ** 0.5
    )


def nearest_embedding(embedding):
    nearest, nearest_distance = None, 1

    for path, embedding2 in embeddings.items():
        distance = cosine_distance(embedding, embedding2)
        if distance < nearest_distance:
            nearest, nearest_distance = path, distance

    return nearest


chat = ChatTemplate(
    {
        "messages": [
            {"role": "system", "content": "You are a Q&A AI."},
            {
                "role": "system",
                "content": "Here are some facts that can help you answer the following question: {{data}}",
            },
            {"role": "user", "content": "{{prompt}}"},
        ]
    }
)


while True:
    prompt = input("user: ")
    if prompt == "exit":
        break

    context = nearest_embedding(get_embedding(prompt))
    data = open(context, "r").read()

    message = chat.completion({"data": data, "prompt": prompt}).choices[0].message

    print(f"{message.role}: {message.content}")
