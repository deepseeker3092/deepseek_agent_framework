Main.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class DeepSeekAgent:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the DeepSeek agent.

        :param model_name: The name of the DeepSeek model on Hugging Face.
        :param device: The device to run the model on (e.g., 'cuda' for GPU, 'cpu' for CPU).
        """
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Error loading the model or tokenizer: {e}")

    def generate_response(self, input_text, max_length=200, temperature=0.7):
        """
        Generate a response based on the input text.

        :param input_text: The input text provided by the user.
        :param max_length: The maximum length of the generated response.
        :param temperature: Controls the randomness of the output. Higher values make the output more random.
        :return: The generated response as a string.
        """
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length, temperature=temperature)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return None


def main():
    agent = DeepSeekAgent()
    while True:
        input_text = input("Please enter your question (type 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        response = agent.generate_response(input_text)
        if response:
            print("Response:", response)


if __name__ == "__main__":
    main()
