from collections import defaultdict
import os
import json
import argparse
import csv
from const import all_langs, all_levels
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_langs", type=str, default=None, help="list of string of languages")
    parser.add_argument("--data_path", type=str, default="./data/text-question/", help="path for writing and reading the data")
    parser.add_argument("--model", type=str, default="chat", help="[chat, gpt4]")
    parser.add_argument("--method", type=str, default="default", help="[default]")
    parser.add_argument("--reasoning", type=str, default="default", help="[default, cot]")
    parser.add_argument("--setting", type=str, default="few-shot", help="[few-shot, zero-shot]")
    return parser.parse_args()


def extract_answer_from_pred(pred: str) -> str:
    """ Extract an answer number using regex from the prediction text"""
    pattern = r'\(([^)]+)\)'
    match = re.search(pattern, pred)
    return match.group(1) if match else None


def compute_acc_score(preds, model):
    """ Compute acc scores for a particular json file """
    match, total, total_time, total_cost = 0, 0, 0, 0
    errors = []
    for question in preds:
        total += 1
        answer = str(question['answer_text']).strip()
        pred = question[model+'_pred'].strip()

        total_time += question['time']
        total_cost += question['cost']

        # prediction of bloom also include the input prompt
        if model == 'bloom':
            pred = pred.replace(question['prompt'], "").strip()
            question['bloom_pred_strip'] = pred

        if len(pred) > 1:
            # Extract the answer number using regex from the prediction text
            # e.g. (1) -> 1
            print('prompt', pred)
            pred = extract_answer_from_pred(pred)
            print('answer', pred)

        if answer == pred:
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

                acc_scores, errors = compute_acc_score(preds, args.model)

                # Modify here to use 'nrQuestions' and 'nrCorrect'
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

                # illformat_file_path = output_folder + f"{lang}-illformat.json"
                # with open(illformat_file_path, 'w') as f:
                #     json.dump(illformats, f)
        
        else:
            print("Cannot find corresponding prediction file!")
    
    result_path = output_folder + 'result.txt'
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
