from collections import defaultdict
import os
import json
import argparse
import csv
from typing import Any
from const import all_langs, all_levels
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from dotenv import load_dotenv
from litellm import completion
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_langs", type=str, default=None, help="list of string of languages")
    parser.add_argument("--data_path", type=str, default="./data/text-question/", help="path for writing and reading the data")
    parser.add_argument("--model", type=str, default="chat", help="[chat, gpt4]")
    parser.add_argument("--method", type=str, default="default", help="[default]")
    parser.add_argument("--reasoning", type=str, default="default", help="[default, cot]")
    parser.add_argument("--setting", type=str, default="few-shot", help="[few-shot, zero-shot]")
    return parser.parse_args()


def before_retry_fn(retry_state: Any) -> None:
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


EVAL_PROMPT = f"""
[TASK]
Assume the role of a teacher who needs to precisely extract the answer from a student's multiple-choice question response written in various languages. When given a task description and a student response:
1. First, understand the core goal outlined in the task description.
2. Next, examine how the student answered the question.
3. Extract the letter corresponding to the student's chosen answer (e.g., "A").
4. Your output should solely consist of this letter, representing the student's answer, ensuring it directly corresponds to the options provided in the question.
5. Ensure accuracy and conciseness in rendering the student’s choice, focusing solely on extracting the correct multiple-choice letter without paraphrasing or additional commentary.

---

[FORMAT]
Follow the following format:

task: a detailed task given to a student which includes the question and formatting instructions
student_response: the student's answer to the given task
extracted_answer: the correct multiple choice option extracted from the student's response; only return the answer, don't add parenthesis

---

[EXAMPLES]
task: Your task: Die volgende is veelvuldige keuse vrae oor Sosiale Wetenskappe.\n\nQuestion you need to solve:\n\n\nDie Nazi se beleid vir Lebensraum van 1939...\n(A) Was 'n ekspansionistiese buitelandse beleid.\n(B) Het Hitler diktatoriale mag gegee\n(C) Het Hitler die Fuhrer gemaak\n(D) Het die Nazi Party die enigste wettige politieke party in Duitsland gemaak.\n\nFormatting instructions: Analiseer eers die vraag wat gevra is in jou eie woorde, vergelyk elke meerkeuse-opsie met die vraag wat gevra is, kies dan die opsie wat die vraag korrek beantwoord. Daarom is die korrekte opsie (X)。\n\nAntwoord:
student_response: (A) Was 'n ekspansionistiese buitelandse beleid
extracted_answer: A
---

[INPUT]
task: {{task}}
student_response: {{student_response}}
extracted_answer: 
"""


@retry(wait=wait_fixed(30), stop=stop_after_attempt(6), before=before_retry_fn)
def get_answer(prompt: str, pred: str) -> str:
    try:
        completions = completion(
            model='azure/gpt-4-turbo',
            messages=[{"role": "user", "content": EVAL_PROMPT.format(task=prompt, student_response=pred)}],
            temperature=0,
        )

        output = completions.choices[0].message.content.strip()
        return output

    except Exception:
        return None


"""
Extracts the answer from a given prediction string.

This function attempts to extract an answer from a prediction string using two methods:
1. It first looks for an answer enclosed in parentheses, expecting the last occurrence to be the answer.
2. If no parentheses are found, it tries to extract a standalone letter or number that is followed by a period.

Args:
    pred (str): The prediction string from which to extract the answer.

Returns:
    str: The extracted answer if found, otherwise None.
"""
def extract_answer_regex(pred: str) -> str:
    # First, try to extract an answer enclosed in parentheses, expecting the last occurrence to be the answer
    match = re.search(r'\(([^)]+)\)[^()]*$', pred)
    if match:
        # Extract only the last alphanumeric portion as the answer (ignoring full sentences or additional content in parentheses)
        inner_content = match.group(1).strip()
        answer_match = re.search(r'\b(\w+)\b\s*$', inner_content)
        if answer_match:
            return answer_match.group(1)
        return inner_content

    # If no parentheses, try to extract a standalone letter or number that is followed by a period
    # Pattern modified to handle answers at the beginning or middle of the string, ending with a period
    match = re.search(r'\b([A-Za-z0-9])\.\s*(?![\w\.])', pred)
    if match:
        return match.group(1)

    return None


def extract_answer_from_pred(prompt: str, pred: str) -> str:
    # try to extract the answer from the prediction
    answer = extract_answer_regex(pred)

    # sometimes this fails, so we use the LLM to extract the output
    if answer is None:
        answer = get_answer(prompt, pred)

    # this could fail too, so we try again
    if answer is not None and len(answer) > 1:
        answer = extract_answer_regex(pred)

    return answer


def add_extracted_answer(preds, model):
    """ Compute acc scores for a particular json file """
    for question in preds:
        pred = question[model+'_pred'].strip()

        if len(pred) > 1:
            # Extract the answer number using regex from the prediction text
            pred = extract_answer_from_pred(question['prompt'], pred)

        question['extracted_answer'] = pred

    return preds


def compute_acc_score(preds):
    """ Compute acc scores for a particular json file """
    match, total, total_time, total_cost = 0, 0, 0, 0
    errors = []
    for question in preds:
        total += 1
        total_time += question['time']
        total_cost += question['cost']

        extracted_answer = question['extracted_answer']

        if question['answer_text'] == extracted_answer:
            match += 1
        else:
            errors.append(question)

    return (total, match, total_time, total_cost), errors


def write_json_to_csv(json_data, output_path):
    if len(json_data) > 1:
        with open(output_path, 'w', newline='') as f:
            fieldnames = json_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in json_data:
                if set(d.keys()).issubset(fieldnames):
                    writer.writerow(d)


def run_evaluate(args, selected_langs):
    acc_dict = defaultdict()
    for lang in selected_langs:
        print('='*50)
        print(f"Run eval on {lang}...")
        output_folder = f"outputs/{args.setting}/{args.reasoning}/{args.method}/model_{args.model}/{lang}/"
        print(output_folder)
        if os.path.exists(output_folder):
            pred_file_path = output_folder + f"{lang}-pred.json"
            if os.path.exists(pred_file_path):
                with open(pred_file_path, "r") as f:
                    preds = json.load(f)

                if ('extracted_answer' not in preds[0]):
                    preds = add_extracted_answer(preds, args.model)
                    with open(pred_file_path, 'w') as f:
                        json.dump(preds, f)

                acc_scores, errors = compute_acc_score(preds)

                acc_dict[lang] = {
                    "nrQuestions": acc_scores[0],
                    "nrCorrect": acc_scores[1],
                    "total_time": acc_scores[2],
                    "total_cost": acc_scores[3],
                    "avg_time": acc_scores[2] / acc_scores[0] if acc_scores[0] > 0 else 0,
                    "avg_cost": acc_scores[3] / acc_scores[0] if acc_scores[0] > 0 else 0
                }

                error_file_path = output_folder + f"{lang}-error.json"
                with open(error_file_path, 'w') as f:
                    json.dump(errors, f)

        else:
            print("Cannot find corresponding prediction file!")
    
    result_path = os.path.join(output_folder, 'result.json')
    print(result_path)
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(acc_dict, f, indent=4)

    print('='*50)
    print('Results:')
    print(json.dumps(acc_dict, indent=4))


def main():
    args = parse_args()
    selected_langs = eval(args.selected_langs) if args.selected_langs else all_langs
    run_evaluate(args, selected_langs)


if __name__ == "__main__":
    main()
