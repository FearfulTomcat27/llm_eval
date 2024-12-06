import re
import textwrap

from openai import OpenAI
from tqdm import tqdm

from data.utils import get_config
from human_eval.data import write_jsonl, read_problems


def construct_problem_prompt(problem):
    return [
        {"role": "system", "content": "You are a professional Python programmer."},
        {
            "role": "user",
            "content": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\nComplete the following Python code without any tests or explanation.Requires that no other functions can be added, and that no libraries other than those in the following code header can be used.\n\n```python\n{problem["prompt"]}\n```\n\n### Response:""",
        },
    ]


def construct_flowchart_prompt(code):
    return [
        {"role": "system", "content": "You're a very helpful assistant."},
        {
            "role": "user",
            "content": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a plantuml flowchart from the following python code.\n{code}\n\n### Response:""",
        },
    ]


def generate_one_completion(client, problem, config):
    response = client.chat.completions.create(
        model=config.get("generate", "model"),
        messages=construct_problem_prompt(problem),
        temperature=config.getfloat("generate", "temperature"),
        top_p=config.getfloat("generate", "top_p"),
        max_tokens=config.getint("generate", "max_tokens"),
    )
    return response.choices[0].message.content


def generate_one_flowchart_completion(client, code, config):
    response = client.chat.completions.create(
        model=config.get("generate", "flowchart_model"),
        messages=construct_flowchart_prompt(code),
        temperature=config.getfloat("generate", "temperature"),
        top_p=config.getfloat("generate", "top_p"),
        max_tokens=config.getint("generate", "max_tokens"),
    )
    return response.choices[0].message.content


def extract_python_code(text, entry_point):
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is None:
        code_block_pattern = re.compile(r"```(?:[Pp]ython\n)?(.*?)```", re.DOTALL)
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        result = code_block.group(1)
        return result

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def extract_flowchart(text):
    code_pattern = re.compile(r"```plantuml\n(.*?)\n```", re.DOTALL)
    code_block = code_pattern.search(text)
    if code_block is not None:
        return code_block.group(1)

    print("No code block found")
    return text


def generate_python_completions(client, problems, config):
    completions = [
        {
            "task_id": task_id,
            "completion": problems[task_id]["prompt"]
            + extract_python_code(
                generate_one_completion(client, problems[task_id], config),
                problems[task_id]["entry_point"],
            ),
        }
        for task_id in tqdm(problems)
    ]
    return completions


def generate_flowchart_completions(client, completions, config):
    flowcharts = [
        {
            "task_id": completion["task_id"],
            "completion": extract_flowchart(
                generate_one_flowchart_completion(
                    client,
                    completion["completion"],
                    config,
                )
            ),
        }
        for completion in tqdm(completions)
    ]
    return flowcharts


def main():
    config = get_config()
    client = OpenAI(
        api_key=config.get("generate", "api_key"),
        base_url=config.get("generate", "base_url"),
    )
    problems = read_problems()
    print("Generating python code completions")
    completions = generate_python_completions(client, problems, config)
    write_jsonl(f"python-{config.get('generate','model')}.jsonl", completions)
    print("Generating flowchart completions")
    flowcharts = generate_flowchart_completions(client, completions, config)
    write_jsonl(f"plantuml-{config.get('generate','model')}.jsonl", flowcharts)


if __name__ == "__main__":
    main()
