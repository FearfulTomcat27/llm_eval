import re

from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm

"""
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

"""


def extract_description(prompt):
    pattern = re.compile(r"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", re.DOTALL)
    block = pattern.search(prompt)

    if block is not None:
        description = block.group(1)
        if description.find(">>>") != -1:
            description = description.split(">>>")[0]

        nlp_description = " ".join([line.strip() for line in description.split("\n")])
        return nlp_description

    raise ValueError("No code block found in prompt")


def main():
    problems = read_problems()
    descriptions = [
        {task_id: extract_description(problems[task_id]["prompt"])}
        for task_id in tqdm(problems)
    ]
    write_jsonl("descriptions.jsonl", descriptions)


if __name__ == "__main__":
    main()
