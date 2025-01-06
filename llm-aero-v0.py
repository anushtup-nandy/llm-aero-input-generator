import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
from openai import OpenAI

class AeroFDocProcessor:
    def __init__(self, base_url: str, api_url: str, api_key: str = None):
        self.base_url = base_url
        self.client = OpenAI(
            base_url=api_url,
            api_key="not-needed"  # LM Studio doesn't require an API key
        )

    def fetch_documentation(self) -> str:
        response = requests.get(self.base_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text

    def parse_html(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')

        examples = []
        structure = {}
        
        # Improved selectors to target code blocks in the Aero-F documentation
        code_blocks = soup.find_all('pre', class_='code')

        for block in code_blocks:
            content = block.get_text().strip()
            if content:
                # Check for keywords to differentiate examples and structure
                if "example" in content.lower() or "input" in content.lower():
                    examples.append(content)
                else:
                    structure[f"section_{len(structure)}"] = content

        return {
            "examples": examples,
            "structure": structure
        }

    def create_training_prompts(self, parsed_data: Dict) -> List[Dict]:
        prompts = []

        for example in parsed_data["examples"]:
            prompts.append({
                "system": "You are an expert in generating aero-f input files. Follow the examples closely.",
                "user": "Generate an aero-f input file based on this example:\n" + example,
                "assistant": example  # Example output (for potential fine-tuning)
            })

        for section, content in parsed_data["structure"].items():
            prompts.append({
                "system": "You are an expert in generating aero-f input files. Adhere to these structural guidelines.",
                "user": f"Generate an aero-f input file following these requirements:\n{content}",
                "assistant": ""  # You might need to manually add example outputs here if needed
            })

        return prompts

    def query_llama(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are an expert in generating aero-f input files."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000  # Adjust as needed
            )
            # Access the generated text correctly
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                return "Error: No response generated or empty response."
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return f"Error: {e}"

    def generate_input_file(self, prompt: str) -> str:
        return self.query_llama(prompt)

def main():
    processor = AeroFDocProcessor(
        base_url="https://frg.bitbucket.io/aero-f/",
        api_url="http://localhost:1234/v1"
    )

    html_content = processor.fetch_documentation()
    parsed_data = processor.parse_html(html_content)

    training_prompts = processor.create_training_prompts(parsed_data)

    with open("training_prompts.json", "w") as f:
        json.dump(training_prompts, f, indent=2)

    example_prompt = "Generate an aero-f input file for a basic linear solver configuration"
    generated_file = processor.generate_input_file(example_prompt)

    print("Generated Input File:")
    print(generated_file)

if __name__ == "__main__":
    main()