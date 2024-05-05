import os
import json
from openai import OpenAI
import pandas as pd
import argparse
import random
import requests
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import concurrent.futures
from dotenv import load_dotenv
from typing import Any, Tuple, List, Dict
from const import all_langs, all_levels, subject2target, answer_word
from litellm import completion

load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=str, default='5', help="Number of samples to test")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers to use")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--selected_langs", type=str, default=None, help="list of string of languages")
    parser.add_argument("--selected_levels", type=str, default=None, help="list of string of levels")
    parser.add_argument("--data_path", type=str, default="./data/text-question/", help="path for writing and reading the data")
    parser.add_argument("--model", type=str, default="chat", help="[gpt-3.5-turbo, gpt-4-turbo, bloom]")
    parser.add_argument("--setting", type=str, default="few-shot", help="[few-shot, zero-shot]")
    parser.add_argument("--reasoning", type=str, default="default", help="[default, cot]")
    parser.add_argument("--method", type=str, default="default", help="[default, en-instruct, en-trans]")
    return parser.parse_args()


def before_retry_fn(retry_state: Any) -> None:
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def parallel_query_llm(args: Tuple[str, ...]) -> str:
    return query_llm(*args)


def parallel_query_bloom_model(args: Tuple[str, ...]) -> str:
    return query_bloom_model(*args)


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_llm(prompt: str, model: str = "gpt-4-turbo", temperature: float = 0) -> str:
    try:
        print('prompt', prompt)
        completions = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()
        print('output', output)

    except Exception as e:
        if "This model's maximum context length is 4097 tokens." in str(e):
            output = "the question is too long"
        else:
            raise e

    return output

@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_bloom_model(api_key: str, prompt: str) -> str:
    model_url = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        response_json = response.json()
        # if the error is due to max context length, save such an error
        if "error" in response_json and response_json['error'].startswith('Input validation error: `inputs`'):
            pred = "the question is too long"
        else:
            raise e

    return pred


def get_answer_word(lang: str, method: str) -> str:
    return answer_word[lang] if method == 'default' else answer_word['english']


def generate_one_example(question: dict, lang: str, method: str, fill_answer: bool = False) -> str:        
    background = '\n'+'\n'.join(question['background_description']) if question['background_description'] != [] else ''

    if method == 'default':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options'])
    elif method == 'en-instruct':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options'])
    elif method == 'en-trans':
        prompt = question['background_description_english'] + '\n' + question['question_text_english'] + '\n' + question['options_english']
    
    if fill_answer:
        answer_word = get_answer_word(lang, method)
        prompt += f'\n{answer_word} {str(question["answer_text"])}'
    
    return prompt


def generate_dev_examples(dev_questions: List[dict], lang: str, method: str) -> Dict[str, Dict[str, List[str]]]:

    # save the dev examples into a dict, according to their levels and subject categories
    dev_example_dict = defaultdict(lambda: defaultdict(list))
    for q in dev_questions:
        level = q['level']
        cate = q['subject_category']
        dev_string = generate_one_example(q, lang, method, fill_answer=True)
        dev_example_dict[level][cate].append(dev_string)
    
    return dev_example_dict


def generate_task_description(test_question: dict, lang: str, method: str) -> str:
    subject = subject2target[lang][test_question['subject_category']]

    # default to use own target language in the prompt/instruction (monolingual setting)
    # for any language, just use english instructions
    if method == 'en-instruct' or method == 'en-trans':
        lang = 'english'

    descriptions = {
        'english': f"The following is a multiple choice question about {subject}.",
        'chinese': f"以下是关于{subject}的单项选择题。",
        'javanese': "Ing ngisor iki ana pitakon pilihan ganda babagan Bahasa Jawa." if test_question['level'] == 'low' else "Menika soal pilihan ganda babagan Bahasa Jawa.",
        'thai': f"ต่อไปนี้เป็นคำถามแบบปรนัย วิชา {subject}.",
        'vietnamese': f"Sau đây là các câu hỏi trắc nghiệm về {subject}.",
        'italian': f"Le seguenti sono domande a risposta multipla su {subject}.",
        'afrikaans': f"Die volgende is veelvuldige keuse vrae oor {subject}.",
        'swahili': f"Yafuatayo ni maswali ya chaguo nyingi kuhusu Kiswahili.",
        'portuguese': f"A seguir estão questões de múltipla escolha sobre {subject}."
    }

    if lang not in descriptions:
        raise NotImplementedError(f"Language '{lang}' not implemented.")

    return 'Your task: ' + descriptions[lang]


def generate_reasoning(reasoning: str, lang: str) -> str:
    # need to instruct the model to only output the option text
    normalHints = {
        'english': 'Please only give the correct option, without any other details or explanations.',
        'chinese': '请仅给出正确选项对应的选项序号而非其他细节。',
        'thai': ' โปรดระบุคำตอบเป็นตัวเลือกที่ถูกต้องโดยไม่ต้องให้รายละเอียดอื่นเพิ่มเติม.',
        'vietnamese': 'Vui lòng chỉ đưa ra phương án đng, không có bất kỳ chi tiết hay giải thích nào khác.',
        'italian': 'Dai solo l\'opzione corretta, senza altri dettagli o spiegazioni',
        'javanese': 'Nyuwun paringaken pilihan wangsulan ingkang leres mawon, tanpa detail utawi penjelasan sanesipun.',
        'afrikaans': 'Gee asseblief net die korrekte opsie, sonder enige ander besonderhede of verduidelikings.',
        'swahili': 'Tafadhali toa chaguo sahihi pekee, bila maelezo yoyote au maelezo.',
        'portuguese': 'Por favor, dê apenas a opção correta, sem quaisquer outros detalhes ou explicaçes.'
    }

    # we want the model to first reason about the question in its own words, 
    # then compare each multiple-choice option with the question asked, 
    # and finally select the option that correctly answers the question
    cotHints = {
        'english': '1) First carefully analyze the question asked in your own words 2) Compare each multiple-choice option with the question asked 3) Then select the option that correctly answers the question. e.g. Therefore, the answer is (X).',
        'chinese': '首先用自己的话仔细分析所提问的问题，将每个选择题选项与所提问的问题进行比较，然后选择正确回答问题的选项。因此，正确的选项是 (X)。',
        'thai': 'วิเคราะห์คำถามที่ถามมาอย่างรอบคอบเป็นคำพูดของคุณเองก่อน, เปรียบเทียบแต่ละตัวเลือกกับคำถามที่ถาม, จากนั้นเลือกตัวเลือกที่ตอบคำถามนั้นได้อย่างถูกต้อง. ดังนั้น, ตัวเลือกที่ถูกต้องคือ (X)。',
        'vietnamese': 'Trước tiên, hãy phân tích câu hỏi được đặt ra bằng cách dùng lời của bạn, so sánh mỗi lựa chọn trắc nghiệm với câu hỏi đã đặt, sau đó chọn lựa chọn trả lời đúng câu hỏi. Vì vậy, lựa chọn đúng là (X)。',
        'italian': 'Prima analizza attentamente la domanda posta con le tue parole, confronta ogni opzione a scelta multipla con la domanda posta, poi seleziona l\'opzione che risponde correttamente alla domanda. Pertanto, l\'opzione corretta è (X)。',
        'javanese': 'Pisanan, tilar mriksa pitakèn sing dipunsuwun kanthi tembung panjenengan dhéwé, bandhingna saben pilihan ganda kalihan pitakèn sing dipunsuwun, banjur pilih pilihan ingkang wangsulan pitakèn punika kanthi leres. Mulane, pilihan sing bener yaiku (X)。',
        'afrikaans': 'Analiseer eers die vraag wat gevra is in jou eie woorde, vergelyk elke meerkeuse-opsie met die vraag wat gevra is, kies dan die opsie wat die vraag korrek beantwoord. Daarom is die korrekte opsie (X)。',
        'swahili': 'Kwanza chambua swali lililoulizwa kwa maneno yako mwenyewe, linganisha kila chaguo la muktadha na swali lililoulizwa, kisha chagua chaguo linalojibu swali hilo kwa usahihi. Hivyo, chaguo sahihi ni (X)。',
        'portuguese': 'Primeiro analise cuidadosamente a questão colocada em suas próprias palavras, compare cada opção de múltipla escolha com a questão feita, depois selecione a opção que responde corretamente à questão. Portanto, a opção correta é (X)。'
    }

    if reasoning == 'default':
        return 'Formatting instructions: ' + normalHints[lang]
    elif reasoning == 'cot':
        return 'Formatting instructions: ' + cotHints[lang]
    else:
        raise NotImplementedError


def example_hints(setting: str, test_question: dict, lang: str, method: str, dev_question: dict, reasoning: str) -> str:
    hint = generate_one_example(test_question, lang, method)
    
    if setting == 'few-shot':
        dev_questions_list = dev_question[test_question['level']][test_question['subject_category']]
        hint = '\n\n'.join(dev_questions_list) + '\n\n' + hint

    return hint


def generate_prompt(lang: str, method: str, setting: str, test_question: dict, dev_question: dict, reasoning: str) -> str:
    task_description = generate_task_description(test_question, lang, method)
    reasoning = generate_reasoning(reasoning, lang)
    hint = example_hints(setting, test_question, lang, method, dev_question, reasoning)
    answer_word = get_answer_word(lang, method)

    prompt = task_description + '\n\n' + hint + '\n\n' + reasoning + f'\n\n{answer_word}'

    print(prompt)

    return prompt


def get_dev_examples(args: argparse.Namespace, lang: str, method: str):
    dev_file_path = args.data_path + f"{lang}-questions-dev.json"
    if os.path.exists(dev_file_path):
        with open(dev_file_path, "r") as f:
            dev_questions = json.load(f)

        return generate_dev_examples(dev_questions, lang, method)
    else:
        raise FileNotFoundError


def file_should_exist(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError


def process_lang(args: argparse.Namespace, lang: str, selected_levels: List[str]) -> None:
    model = args.model
    method = args.method
    setting = args.setting
    reasoning = args.reasoning

    output_folder = f"outputs/{setting}/{reasoning}/{method}/model_{model}/{lang}/"
    os.makedirs(output_folder, exist_ok=True)

    dev_examples = get_dev_examples(args, lang, method) if setting == 'few-shot' else {}

    test_file_path = args.data_path + f"{lang}-questions-test.json"
    file_should_exist(test_file_path)

    with open(test_file_path, "r") as f:
        test_questions = json.load(f)

    # only take certain number of examples to test
    if args.num_samples != 'all':
        num_samples = int(args.num_samples)
        test_questions = test_questions[:num_samples]
    
    # if only want to test on certain levels
    if len(selected_levels) < 3:
        test_questions = [q for q in test_questions if q['level'] in selected_levels]

    # generate prompts
    all_prompts: List[str] = []

    for question in test_questions:
        prompt = generate_prompt(lang, method, setting, question, dev_examples, reasoning)
        all_prompts.append(prompt)
    
    # inference in batch
    prompt_args = [(p, model) for p in all_prompts]
    
    if model not in ["bloom"]:
        parallel_call = parallel_query_llm
    elif model == "bloom":
        parallel_call = parallel_query_bloom_model

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        predictions = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

    # save the predictions
    for idx, question in enumerate(test_questions):
        question[model+'_pred'] = predictions[idx]    # save the pred
        question['prompt'] = all_prompts[idx]         # also save the prompt
    
    with open(f"{output_folder}/{lang}-pred.json", "w") as f:
        json.dump(test_questions, f)
    
    print(f"Done: {len(test_questions)} {lang} questions!")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    selected_langs = eval(args.selected_langs) if args.selected_langs else all_langs
    selected_levels = eval(args.selected_levels) if args.selected_levels else all_levels

    for lang in selected_langs:
        process_lang(args, lang, selected_levels)


if __name__ == "__main__":
    main()