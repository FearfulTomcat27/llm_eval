import configparser
import json
import re

import datasets
from case_convert import camel_case

DATASET_REVISION = "8a4cb75204eb3d5855a81778db6b95bfc80c9136"


def stream_jsonl(file):
    with open(file, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)


def read_jsonl(file) -> list:
    data = stream_jsonl(file)
    return list(data)


def get_humaneval_task_id(task_id):
    return f"HumanEval/{task_id.split('_')[1]}"


def load_datasets(language) -> list:
    problems = datasets.load_dataset(
        "nuprl/MultiPL-E",
        f"humaneval-{language}",
        revision=DATASET_REVISION,
        split="test",
    )
    return list(problems)


def get_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


def get_entry_point_from_multiple(task_id: str):
    function_name = re.sub(r"HumanEval_\d+_", "", task_id)
    return camel_case(function_name)


def extract_task_id(name):
    return name.split("_")[0] + "/" + name.split("_")[1]


def extract_function_declare(problem):
    pattern = re.compile(r"\s+(public\sstatic.*?)\s\{", re.DOTALL)
    match = pattern.search(problem["prompt"])
    if match is not None:
        return match.group(1)
    else:
        raise ValueError(f"Function declaration not found in {problem['name']}")


def extract_description(text):
    lines = [
        line.strip().lstrip("// ")
        for line in re.search(
            rf"(// .*?\n)\s+(?://\s>>>.*?)?\s+public\sstatic",
            text,
            re.DOTALL,
        )
        .group(1)
        .split("\n")
    ]
    return " ".join(lines)
