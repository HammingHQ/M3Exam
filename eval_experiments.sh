#!/bin/bash

settings=('zero-shot' 'few-shot')
models=('azure/gpt-35-turbo' 'azure/gpt-4-turbo' 'groq/Llama3-70b-8192' 'gemini/gemini-pro' 'command-r' 'command-r-plus' 'claude-3-opus-20240229' 'claude-3-sonnet-20240229' 'claude-3-haiku-20240307' 'gpt-4o')
reasoning=('default' 'cot')
num_samples=1

# loop through all combinations of settings, models, reasoning
for setting in ${settings[@]}; do
    for model in ${models[@]}; do
        for reasoning in ${reasoning[@]}; do
            python eval.py --setting ${setting} --model ${model} --reasoning ${reasoning}
        done
    done
done
