from transformers import AutoTokenizer, AutoModelForCausalLM
import random


class TextGenerator:
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        prompt_length = len(self.tokenizer.decode(inputs[0]))
        outputs = self.model.generate(
            inputs,
            max_length=500,
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


class PromptGenerator:
    def __init__(self):
        self.facts_in_prompt = 5
        self.max_new_facts = 9
        self.LEAD = 'I know all there is to know about birds. I will tell you some cool stuff about birds.'
        self.CORE_FACTS = [
            'Bird Fact: Did you know that the Sword-billed Hummingbird is the only bird with a bill longer than its body?',
            'Bird Fact: Did you know that owls cannot swivel their eyes? Instead they move their heads completely around to see straight behind them.',
            'Bird Fact: Did you know that the only bird with nostrils at the end of its beak is the kiwi? This placement helps it sniff for food, such as worms and insects on the ground.',
            'Bird Fact: Mockingbirds can imitate many sounds, from a squeaking door to a cat meowing.',
            'Bird Fact: Chickens that lay brown eggs have red ear lobes. There is a genetic link between the two.',
            'Bird Fact: Crows have the largest cerebral hemispheres (brains), relative to body size, of any avian family.',
            'Bird Fact: The human ear, in fact, is a machine to hear birds.',
            'Bird Fact: A bird\'s head is shaped like a cow\'s head.',
            'Bird Fact: A bird cannot fly.',
        ]
        self.new_facts = []

    def __call__(self):
        facts = list()
        facts.extend(self.CORE_FACTS)
        facts.extend(self.new_facts)
        facts = random.sample(facts, k=self.facts_in_prompt)
        random.shuffle(facts)
        return self.LEAD + '\n' + '\n'.join(facts)

    def set_new_facts(self, new_facts):
        self.new_facts = new_facts

    def add_new_facts(self, new_facts):
        self.new_facts.extend(new_facts)
        if len(new_facts) > self.max_new_facts:
            self.new_facts = random.sample(self.new_facts, k=self.max_new_facts)