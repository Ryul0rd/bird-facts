import os
os.environ['TRANSFORMERS_CACHE'] = '/src/models/'

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import random
import re
import argparse


class PromptGenerator:
    def __init__(self):
        f = open("src/seed_facts.txt","r")
        lines = f.readlines()
        f.close()
        self.LEAD = lines[0]
        self.FACTS = lines[1:]
        self.FACTS_IN_PROMPT = 5

    def __call__(self):
        prompt_facts = random.sample(self.FACTS, k=self.FACTS_IN_PROMPT)
        random.shuffle(prompt_facts)
        prompt = self.LEAD
        for idx, fact in enumerate(prompt_facts):
            prompt += f'{idx + 1}. {fact}' 
        return prompt


class TextGenerator:
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        prompt_length = len(self.tokenizer.decode(inputs[0]))
        outputs = self.model.generate(
            inputs,
            max_length=300,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=60,
            #num_beams=3,
            no_repeat_ngram_size=8,
            pad_token_id=50256, # Set pad_token_id to eos_token_id so we don't see it on the CLI
            )
        generated = self.tokenizer.decode(outputs[0])[prompt_length:]
        return generated


parser = argparse.ArgumentParser()
parser.add_argument('--n_facts', type=int, default=10, help='The number of facts to be generated.')
args = parser.parse_args()

checkpoint = 'EleutherAI/gpt-neo-1.3B'
text_gen = TextGenerator(checkpoint=checkpoint)
prompt_gen = PromptGenerator()
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
topic = 'birds'
threshold = 0.5

fact_count = 0
n_facts = 10
while fact_count < args.n_facts:
    prompt = prompt_gen()
    facts = text_gen(prompt).split('\n')[:-1]

    facts = [fact for fact in facts if len(fact) > 20]
    facts = [fact for fact in facts if len(re.findall('^[0-9]+\. ', fact)) > 0]
    facts = [re.sub('^[0-9]+\. ', '', fact) for fact in facts]

    facts = classifier(facts, topic)
    facts = [fact for fact in facts if fact['scores'][0] > threshold]
    facts = [fact['sequence'] for fact in facts]

    for fact in facts:
        fact_count += 1
        print(fact)