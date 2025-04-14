import copy
import json
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if os.environ["OPENAI_API_KEY"] is None:
    raise Exception("OPENAI_API_KEY not set")


def insert_params(string, **kwargs):
    pattern = r"{{(.*?)}}"
    matches = re.findall(pattern, string)
    for match in matches:
        replacement = kwargs.get(match.strip())
        if replacement is not None:
            string = string.replace("{{" + match + "}}", replacement)
    return string


class ChatTemplate:
    def __init__(self, template):
        self.template = template

    def from_file(template_file):
        with open(template_file, "r") as f:
            template = json.load(f)
        return ChatTemplate(template)

    def completion(self, parameters):
        instance = copy.deepcopy(self.template)
        for item in instance["messages"]:
            item["content"] = insert_params(item["content"], **parameters)
        return client.chat.completions.create(model="gpt-3.5-turbo", **instance)
