import argparse
import json
import re
import requests
import textwrap
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="Model api address, Recommended Http.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, reference model name in ollama.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Model params for temperature."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Model params for top_p."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Generate k completions for one problem,for calculate pass@k.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="samples.jsonl",
        help="The output file name, jsonl required",
    )
    parser.add_argument(
        "--write-",
        type=bool,
        default=False,
        help="Write to the file after all the results have been generated.",
    )
    return parser.parse_args()


def generate_one_completion(args, prompt: str):
    payload = json.dumps(
        {
            "model": args.model,
            "system": "Write code in Python that meets the problem following. Ensure that the code you write is efficient, readable. Remember, do not need to explain the code you wrote.",
            "prompt": prompt,
            "stream": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    response = requests.request("POST", args.api_url, headers=headers, data=payload)
    data = response.json()
    return data["response"]


def extract_code_from_completion(text: str, entry_point: str):
    # 正则表达式匹配代码块
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

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def write_after_generate(args, problems):
    samples = [
        dict(
            task_id=task_id,
            completion=extract_code_from_completion(
                generate_one_completion(args, problems[task_id]["prompt"]),
                problems[task_id]["entry_point"],
            ),
        )
        for task_id in problems
        for _ in range(args.k)
    ]
    write_jsonl(args.output_file, samples)


def write_while_generate(args, problems):
    with open(args.output_file, "wb") as fp:
        for task_id in tqdm(problems):
            for _ in range(args.k):
                task = dict(
                    task_id=task_id,
                    completion=extract_code_from_completion(
                        generate_one_completion(args, problems[task_id]["prompt"]),
                        problems[task_id]["entry_point"],
                    ),
                )
                fp.write((json.dumps(task) + "\n").encode("utf-8"))


if __name__ == "__main__":
    args = parse_args()
    problems = read_problems()
    if args.wag:
        write_after_generate(args, problems)
    else:
        write_while_generate(args, problems)
