import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Tuple
import gradio as gr
from openai import OpenAI
import re
import yaml

class KnowledgeEngineer:
    def __init__(self, base_url: str, api_url: str):
        self.base_url = base_url
        self.client = OpenAI(
            base_url=api_url,
            api_key="not-needed"
        )

    def fetch_documentation(self) -> str:
        response = requests.get(self.base_url)
        response.raise_for_status()
        return response.text

    def extract_relevant_html(self, html_content: str) -> str:
        """
        Extracts the most relevant sections from the HTML documentation using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Select sections related to input file structure, parameters, etc.
        # You'll need to adjust these selectors based on the actual Aero-F HTML
        relevant_sections = soup.find_all(['h3', 'h4', 'h5', 'pre'])

        extracted_html = ""
        for section in relevant_sections:
            extracted_html += str(section)

        return extracted_html

    def generate_knowledge_base_prompt(self, html_content: str) -> str:
        """
        Generates a prompt for the LLM to create a structured knowledge base.
        Now uses extracted and summarized HTML content.
        """
        prompt = f"""
        You are a knowledge engineer tasked with creating a comprehensive and structured knowledge base from the Aero-F solver documentation.

        Here are the most relevant sections of the HTML documentation:

        ```html
        {html_content}
        ```

        Your task is to:

        1. Identify all key sections, parameters, and concepts related to Aero-F input files.
        2. Extract parameter names, data types (e.g., string, integer, float, list), descriptions, and possible values (if specified or can be inferred).
        3. Determine the hierarchical relationships between parameters. Identify which parameters are grouped under others using the 'under' keyword.
        4. If possible, extract any constraints or rules related to the parameters (e.g., valid ranges, dependencies between parameters).
        5. Represent this information in a structured JSON format. The format should be easy to parse programmatically and should capture the hierarchical nature of the Aero-F input file structure.

        Example of the desired output structure (JSON):

        ```json
        {{
          "Problem": {{
            "Type": {{
              "description": "Type of problem (Steady, Unsteady, ...)",
              "values": [
                "Steady",
                "Unsteady"
              ]
            }},
            "Mode": {{
              "description": "Mode of operation (Dimensional, NonDimensional)",
              "values": [
                "Dimensional",
                "NonDimensional"
              ]
            }}
          }},
          "Input": {{
            "Prefix": {{
              "description": "Prefix for input files",
              "type": "string"
            }}
          }},
          "Output": {{
            "under Postpro": {{
              "Prefix": {{
                "description": "Prefix for output files",
                "type": "string"
              }}
            }}
          }}
        }}
        ```

        Create a detailed and accurate knowledge base that can be used by another agent to generate Aero-F input files.
        """
        return prompt

    def parse_knowledge_base(self, kb_text: str) -> Dict:
        """
        Parses the knowledge base text (now expecting JSON format) into a Python dictionary.
        Includes error handling and a simple validation check.
        """
        try:
            knowledge_base = json.loads(kb_text)

            # Basic validation: Check if the result is a dictionary
            if not isinstance(knowledge_base, dict):
                raise ValueError("Parsed knowledge base is not a dictionary")

            return knowledge_base
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}
        except ValueError as e:
            print(f"Validation error: {e}")
            return {}

    def create_knowledge_base(self):
        """
        Fetches the documentation, extracts relevant sections, prompts the LLM, and parses the result.
        """
        html_content = self.fetch_documentation()
        extracted_html = self.extract_relevant_html(html_content)
        prompt = self.generate_knowledge_base_prompt(extracted_html)
        kb_text = self.query_llama(prompt)
        knowledge_base = self.parse_knowledge_base(kb_text)

        # Save the knowledge base to a file (now as JSON)
        with open("knowledge_base.json", "w") as f:
            json.dump(knowledge_base, f, indent=2)

        return knowledge_base

    def query_llama(self, prompt: str) -> str:
        # Same query_llama function as before
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are an expert in understanding technical documentation and creating structured knowledge bases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000  # Increased max_tokens for potentially larger knowledge base
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                return "Error: No response generated or empty response."
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return f"Error: {e}"

class InputFileGenerator:
    def __init__(self, api_url: str, knowledge_base_file: str = "knowledge_base.json"):
        self.client = OpenAI(
            base_url=api_url,
            api_key="not-needed"
        )
        self.knowledge_base = self.load_knowledge_base(knowledge_base_file)
        self.base_url = "https://frg.bitbucket.io/aero-f/"

    def load_knowledge_base(self, knowledge_base_file: str) -> Dict:
        """Loads the knowledge base from a JSON file."""
        try:
            with open(knowledge_base_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Knowledge base file not found at {knowledge_base_file}")
            return {}

    def fetch_documentation(self) -> str:
        response = requests.get(self.base_url)
        response.raise_for_status()
        return response.text

    def parse_html(self, html_content: str) -> Tuple[List[Dict], Dict]:
        soup = BeautifulSoup(html_content, 'html.parser')

        examples = []
        structure = {}

        # Find the "Examples" section
        examples_section = soup.find('h2', string='5 EXAMPLES')

        if examples_section:
            # Find all example blocks and their preceding headings
            example_blocks = examples_section.find_next_siblings()

            current_example = ""
            current_heading = ""
            for element in example_blocks:
                if element.name in ['h4', 'h5', 'h6', 'h7', 'h8', 'h9']:
                    if current_example:
                        examples.append({"heading": current_heading, "content": current_example})
                        current_example = ""
                    current_heading = element.get_text().strip()
                elif element.name == 'pre' and 'code' in element.get('class', []):
                    current_example += element.get_text().strip() + "\n"
                elif current_example:
                    examples.append({"heading": current_heading, "content": current_example})
                    current_example = ""
                    current_heading = ""

            if current_example:
                examples.append({"heading": current_heading, "content": current_example})
        else:
            print("Warning: 'Examples' section not found in the HTML.")

        return examples, structure

    def generate_input_file_prompt(self, user_prompt: str, examples: List[Dict]) -> str:
        # Enhanced prompt using the knowledge base
        examples_text = ""
        for example in examples:
            examples_text += f"\n\nExample ({example['heading']}):\n{example['content']}"

        # Use knowledge base to create a structured representation of parameters
        parameter_section = self.format_knowledge_base_for_prompt(self.knowledge_base)

        # Build the final prompt
        full_prompt = f"""You are an expert in generating aero-f input files. You have access to a knowledge base that describes the structure and parameters of Aero-F input files.

        Here is the knowledge base in JSON format:

        ```json
        {parameter_section}
        ```

        Here are some examples of aero-f input files, pay very close attention to the use of 'under' to create a hierarchical structure:

        {examples_text}

        The user has provided the following request:

        "{user_prompt}"

        Based on this request, the knowledge base, and the examples, generate an aero-f input file.

        Use the knowledge base to understand the meaning and valid values of parameters mentioned in the user prompt. 
        Make sure to use the 'under' keyword to create a structure like in the examples. Follow the structure and parameter descriptions in the knowledge base closely.
        """

        return full_prompt
    
    def format_knowledge_base_for_prompt(self, knowledge_base: Dict, indent_level=0) -> str:
        """Formats the knowledge base into a string suitable for the prompt."""
        return json.dumps(knowledge_base, indent=2)

    def query_llama(self, prompt: str) -> str:
        # Same query_llama function as before
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are an expert in generating aero-f input files."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                return "Error: No response generated or empty response."
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return f"Error: {e}"

    def generate_input_file(self, user_prompt: str, examples: List[Dict]) -> str:
        prompt = self.generate_input_file_prompt(user_prompt, examples)
        return self.query_llama(prompt)

def generate_aero_f_input(user_prompt):
    # Use the InputFileGenerator agent
    generator = InputFileGenerator(
        api_url="http://localhost:1234/v1"
    )

    processor = AeroFDocProcessor(
        base_url="https://frg.bitbucket.io/aero-f/",
        api_url="http://localhost:1234/v1"
    )

    html_content = processor.fetch_documentation()
    examples, structure = processor.parse_html(html_content)

    generated_file_content = generator.generate_input_file(user_prompt, examples)
    return generated_file_content

if __name__ == "__main__":
    # 1. Create the knowledge base using the KnowledgeEngineer agent
    engineer = KnowledgeEngineer(
        base_url="https://frg.bitbucket.io/aero-f/",
        api_url="http://localhost:1234/v1"
    )
    knowledge_base = engineer.create_knowledge_base()
    print("Knowledge base created and saved to knowledge_base.json")

    # 2. Launch the Gradio interface for input file generation
    iface = gr.Interface(
        fn=generate_aero_f_input,
        inputs=[
            gr.Textbox(label="User Prompt", placeholder="Enter a detailed description of the desired simulation...")
        ],
        outputs=gr.Textbox(label="Generated Aero-F Input File", lines=10),
        title="Aero-F Input File Generator",
        description="Generate Aero-F input files using a local Llama model, with a knowledge base and examples."
    )
    iface.launch()