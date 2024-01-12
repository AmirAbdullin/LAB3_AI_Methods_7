from dataclasses import dataclass

from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline


@dataclass
class LLM:
    def __post_init__(self) -> None:
        self.device = "cpu"
        self.config = {
            "max_length": 200,
            "temperature": 1.1,
            "top_p": 2.0,
            "num_beams": 10,
            "repetition_penalty": 1.5,
            "num_return_sequences": 9,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
        }

        model_name = "ai-forever/rugpt3small_based_on_gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)

    def generate(self, prompt: str) -> str:
        # https://github.com/ai-forever/ru-gpts#usage-1
        output = self.generation(prompt, **self.config)[0]["generated_text"]
        return output



