from human_eval.data import read_problems, write_jsonl
from openai import OpenAI
from tqdm import tqdm

from data.utils import (
    load_datasets,
    extract_description,
    get_entry_point_from_multiple,
    extract_function_declare,
    extract_task_id,
    get_config,
)
from eval_multiple import extract_java_code
from gen_mermaid import extract_python_code


def construct_prompt(
    user_content: str,
    system_content: str = "You are a helpful assistant.",
):
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": user_content},
    ]


def generate_completion(
    client,
    messages: list,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def generate_python_code(
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    k: int,
):
    problems = read_problems()
    results = {
        task_id: problems[task_id]["prompt"]
        + extract_python_code(
            generate_completion(
                client,
                construct_prompt_for_python_generation(problems[task_id]["prompt"]),
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
            problems[task_id]["entry_point"],
        )
        for _ in range(k)
        for task_id in tqdm(problems)
    }
    return results


def construct_prompt_for_python_generation(problem: str):
    system_content = "You are a very experienced python programmer."
    user_content = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following python code without any tests or explanation\n{problem}\n\n### Response:"""
    return construct_prompt(user_content, system_content)


def construct_prompt_with_code_without_context(
    requirement: str,
    function_name: str,
    function_declare: str,
    source_code: list[dict],
    language: str,
):
    user_content = f"""Below is a an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a static {function_name} function using {language}. And I will give you the Code of Python. Require satisfy the following requests.\n{requirement}\n\n### Python:\n{source_code}\n\nThe function is declared as follows.\n{function_declare}\n\n### Response:\n"""
    system_content = f"You are a very experienced {language} programmer."
    return construct_prompt(user_content, system_content)


def generate_java_code(
    client: OpenAI,
    source_codes: dict,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    k: int,
):
    problems = load_datasets("java")
    results = []
    for problem in tqdm(problems):
        requirement = extract_description(problem["prompt"])
        function_name = get_entry_point_from_multiple(problem["name"])
        function_declare = extract_function_declare(problem)
        completions = [
            extract_java_code(
                generate_completion(
                    client,
                    construct_prompt_with_code_without_context(
                        requirement,
                        function_name,
                        function_declare,
                        source_codes[extract_task_id(problem["name"])],
                        "java",
                    ),
                    model,
                    temperature,
                    top_p,
                    max_tokens,
                ),
                problem["name"],
            )
            for _ in range(k)
        ]
        task = {
            "name": problem["name"],
            "language": "java",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "prompt": problem["prompt"],
            "tests": problem["tests"],
            "completions": completions,
            "stop_tokens": problem["stop_tokens"],
        }
        results.append(task)
    return results


def main():
    config = get_config()

    # ===== Declare variables =====
    client = OpenAI(
        api_key=config.get("generate", "api_key"),
        base_url=config.get("generate", "base_url"),
    )
    model = config.get("generate", "model")
    language = config.get("generate", "language")
    temperature = config.getfloat("generate", "temperature")
    top_p = config.getfloat("generate", "top_p")
    max_tokens = config.getint("generate", "max_tokens")
    k = config.getint("generate", "k")
    optimize = config.getboolean("generate", "optimize")
    output_file = f"{model}-{language}-{'optimize' if optimize else 'baseline'}.jsonl"

    # ===== Start Generating source code of python using HumanEval =====
    print("Generating python completions...")
    source_codes = generate_python_code(
        client, model, temperature, top_p, max_tokens, k
    )

    # ===== Start Generating source code of java or others using MultiPL-E =====
    print(f"Generating {language} completions...")
    source_codes_java = generate_java_code(
        client, source_codes, model, temperature, top_p, max_tokens, k
    )
    write_jsonl(output_file, source_codes_java)


if __name__ == "__main__":
    main()
