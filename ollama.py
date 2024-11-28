import gzip
import json
import re

from human_eval.data import read_problems
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="http://10.16.48.223:11434/v1", api_key="ollama")


def get_function_name(prompt, entry_point):
    pattern = re.compile(rf"(def\s+{entry_point}.*?:\n).*", re.DOTALL)
    code = pattern.search(prompt)
    if code is not None:
        return code.group(1)
    else:
        raise ValueError("Code block not found")


def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def get_one_completion(code):
    response = client.chat.completions.create(
        model="qwen2.5-coder:7b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You're a very experienced Python programmer.",
            },
            {
                "role": "user",
                "content": f"""Below is a python code. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a plantuml flowchart from the following python code\n{code}\n\n### Response:""",
            },
        ],
        temperature=0.2,
        top_p=0.95,
    )
    return response.choices[0].message.content


def extract_mermaid(text):
    code_pattern = re.compile(r"```plantuml\n(.*?)\n```", re.DOTALL)
    code_block = code_pattern.search(text)
    if code_block is not None:
        return code_block.group(1)

    raise ValueError("Code block not found")


problems = read_problems()
completions = list(stream_jsonl("samples.jsonl"))

data = []
with open("plantuml.jsonl", "wb") as fp:
    for completion in tqdm(completions):
        problem = problems[completion["task_id"]]
        solution = (
            get_function_name(problem["prompt"], problem["entry_point"])
            + "\n"
            + completion["completion"]
        )
        res = get_one_completion(solution)
        item = {"task_id": completion["task_id"], "syntax": extract_mermaid(res)}
        fp.write((json.dumps(item) + "\n").encode("utf-8"))
