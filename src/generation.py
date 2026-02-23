import argparse
import os
import re
import json

from tqdm import tqdm

from models import LLMModel

from datasets import load_dataset
import json

EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>"]


def truncate_after_eof_strings(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)

    if match:
        return text[:match.start()]
    else:
        return text


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def generate_prompt(problem):
    prompt = "\nQUESTION:\n"
    prompt += problem["question"]
    starter_code = problem["starter_code"] if len(
        problem.get("starter_code", [])) > 0 else None

    fn_name = problem["fn_name"]

    if starter_code:
        prompt += starter_code
    if (not fn_name) and (not starter_code):
        call_format = "\nPlease write your code using Standard Input, i.e. input() and print()."
        prompt += call_format
    else:
        call_format = "\Please write your code using Call-Based format."
        prompt += call_format

    if starter_code:
        prompt += f"The starter code is provided as below. Please finish the code.\n<starter_coder>{starter_code}<starter_coder>\n"

    return prompt


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    match args.model:
        case "codellama_7b":
            model = LLMModel(name='codellama_7b',
                             model_id='codellama/CodeLlama-7b-Instruct-hf',
                             prompt_template='<s>[INST] {prompt} [/INST]',
                             extract_pattern=r'```([\s\S]*?)```')
        case "deepseekcoder":
            model = LLMModel(
                backend="ollama",
                name='deepseek_coder_6_7b',
                model_id='deepseek-ai/deepseek-coder-6.7b-instruct',
                prompt_template=
                '''You are an AI programming assistant, utilizing the DeepSeek Coder model.\n### Instruction:\n{prompt}\n### Response:\n''',
                extract_pattern=r'### Response:.*?```python(.*?)```')
        case "gemma":
            model = LLMModel(
                backend="ollama",
                name='gemma_7b',
                model_id='google/gemma-7b',
                access_token='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',
                prompt_template='{prompt}',
                extract_pattern=r'ANSWER:(.*?)<eos>')
        case "mixtral_8x_7b":
            model = LLMModel(
                backend="ollama",
                name='mixtral_8x_7B',
                model_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
                access_token='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',
                prompt_template='<s>[INST] {prompt} [/INST] ',
                extract_pattern=r'```python(.*?)```')
        case "mixtral_7b":
            model = LLMModel(
                backend="ollama",
                name='mixtral_7B',
                model_id='mistralai/Mistral-7B-Instruct-v0.2',
                access_token='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',
                prompt_template='<s>[INST] {prompt} [/INST] ',
                extract_pattern=r'```python(.*?)```')
        case "llama2":
            model = LLMModel(backend="ollama",
                             name='llama2',
                             model_id='meta-llama/Llama-2-7b-chat-hf',
                             prompt_template='<s>[INST] {prompt} [/INST] ',
                             extract_pattern=r'```([\s\S]*?)```')
        case "llama3":
            model = LLMModel(backend="ollama",
                             name='llama3',
                             model_id='meta-llama/Meta-Llama-3-8B-Instruct',
                             prompt_template='{prompt}',
                             extract_pattern=r'```python(.*?)```')
        case _:
            raise ValueError(f"Unknown model {args.model}")


#     problems = load_problems(args.problems_root)
# problems = load_dataset("codeparrot/apps", split="test")
    problems = read_json(args.data_path)
    # print(problems)
    print("length: ", len(problems))

    to_generate = {p["id"]: args.n for p in problems}
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as file:
            existing_results = [
                json.loads(item) for item in file.read().strip().splitlines()
            ]
            # Filtering logic
            for result in existing_results:
                to_generate[result["id"]] -= 1

    for problem in tqdm(problems):
        for run_id in reversed(range(1, to_generate[problem["id"]] + 1)):
            prompt = generate_prompt(problem)

            response = None
            attempts = 0
            # Try for up to 5 times to generate a response
            while response is None and attempts < 5:

                response, prompt_new = model.generate(
                    prompt, temperature=args.temperature)

                attempts += 1
            print('----------------------------------------\n', prompt_new,
                  '\n---------------------------------------------')
            if not response:
                print(f"Failed to generate for problem {problem['task_id']}")
                break  # Break out of the run_id loop to move on to the next problem

            with open(args.save_path, 'a') as file:
                json.dump(
                    {
                        "task_id": problem["task_id"],
                        "run_id": run_id,
                        "prompt": prompt_new,
                        "input": problem['input'],
                        "output": problem['output'],
                        "deal_response": model.extract_code(response),
                        "full_response": response
                    },
                    file  # No indents!
                )
                file.write(
                    '\n')  # Add a newline for separation between entries
            response = model.extract_code(response)
            print('----------------------------------------\n', response,
                  '\n---------------------------------------------')
            to_generate[problem["task_id"]] -= 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate code for problems.")
    parser.add_argument("--model",
                        type=str,
                        default="gpt4",
                        choices=[
                            "gpt4",
                            "gpt3.5",
                            "gemini_pro",
                            "codellama_7b",
                            "deepseekcoder",
                            "wizardcoder",
                            "starcoder",
                            "magiccoder",
                            "chatglm",
                            "qwen",
                            "wenxin",
                            "codegeex",
                            "gemma",
                            "mixtral_8x_7b",
                            "mixtral_7b",
                            "starcoder2",
                            "llama2",
                            "claude3",
                            "llama3",
                        ],
                        help="Model to use for generation.")

    #     parser.add_argument("--problems_root", type=str, default="../mmcode_dataset", help="Path to the root directory of problems.")
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--save_path",
                        type=str,
                        help="Path where the results will be saved.")
    parser.add_argument("--data_path", type=str, help="Path of dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generations per problem.")

    parser.add_argument("--temperature",
                        type=float,
                        default=0.001,
                        help="The temperature used in the generation.")
    args = parser.parse_args()
    main(args)
