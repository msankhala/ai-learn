import json
import os
from get_embedding import get_embedding

embeddings = {}

for f in os.listdir("./racing"):
    path = os.path.join("./racing", f)
    with open(path, "r") as f:
        text = f.read()

    embeddings[path] = get_embedding(text)

with open("embeddings.json", "w+") as f:
    json.dump(embeddings, f)
