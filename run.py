from helpers import PromptGenerator, TextGenerator
from transformers import pipeline

def main():
    threshold = 0.5 # This is how confident our classifier should need to be for a fact to pass.

    #checkpoint = 'gpt2'                    # Uses aprox 1 GB RAM
    #checkpoint = 'gpt2-large'              # Uses aprox 6 GB RAM
    checkpoint = 'EleutherAI/gpt-neo-1.3B' # Uses aprox 10 GB RAM
    #checkpoint = 'EleutherAI/gpt-neo-2.7B' # Uses aprox 20 GB RAM

    prompt_gen = PromptGenerator()
    text_gen = TextGenerator(checkpoint=checkpoint)
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    topic = 'birds'

    prompt = prompt_gen()
    while True:
        # Create our facts
        facts = text_gen(prompt).split('\n')

        # Basic cleaning
        if len(facts) > 0:
            facts = facts[:-1] # Remove the sometimes incomplete final line. Should really check to see if it's complete instead.
        else:
            prompt = prompt_gen()
            continue
        facts = list(filter(lambda fact: len(fact) > 20, facts)) # Remove overly short lines
        facts = list(filter(lambda fact: 'Bird Fact: ' in fact, facts)) # Lines that don't start with 'Bird Fact: ' are generally of low quality
        facts = list(map(lambda fact: fact.replace('Bird Fact: ', ''), facts)) # Hide 'Bird Fact: ' from our classifier

        # Checking to see if our facts are about birds
        facts = classifier(facts, topic)
        facts = list(filter(lambda fact: fact['scores'][0] > threshold, facts))
        facts = list(map(lambda fact: fact['sequence'], facts))

        # Printing results and getting our next prompt
        facts = list(map(lambda fact: 'Bird Fact: ' + fact, facts))
        for fact in facts:
            print(fact)
        prompt_gen.add_new_facts(new_facts=facts)
        prompt = prompt_gen()

if __name__ == '__main__':
    main()