import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Tuple
import gradio as gr
from openai import OpenAI
import re
import yaml

class AeroFDocProcessor:
    def __init__(self, base_url: str, api_url: str, knowledge_base_file: str = "knowledge_base.yaml"):
        self.base_url = base_url
        self.client = OpenAI(
            base_url=api_url,
            api_key="not-needed"
        )
        self.knowledge_base = self.load_knowledge_base(knowledge_base_file)

    def load_knowledge_base(self, knowledge_base_file: str) -> Dict:
        """Loads the knowledge base from a YAML file."""
        try:
            with open(knowledge_base_file, "r") as f:
                return yaml.safe_load(f)
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
                if element.name in ['h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9']:
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

        # Parse examples to build tree-like structure
        structured_examples = []
        for example in examples:
            structured_examples.append({
                "heading": example["heading"],
                "structure": self.parse_aero_f_example(example["content"])
            })

        return structured_examples, structure

    def parse_aero_f_example(self, example_text: str) -> Dict:
        """Recursively parses an Aero-F example to build a tree-like structure."""
        structure = {}
        lines = example_text.split("\n")
        i = 0

        def parse_section(name: str, lines: List[str], i: int) -> Tuple[Dict, int]:
            section = {}
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("under"):
                    match = re.match(r"under\s+(\w+)\s*\{", line)
                    if match:
                        sub_section_name = match.group(1)
                        sub_section, i = parse_section(sub_section_name, lines, i + 1)
                        section[f"under {sub_section_name}"] = sub_section
                elif line.endswith("}"):
                    return section, i + 1
                elif "=" in line:
                    param, value = line.split("=", 1)
                    section[param.strip()] = value.strip().rstrip(";")
                i += 1
            return section, i

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("under"):
                match = re.match(r"under\s+(\w+)\s*\{", line)
                if match:
                    section_name = match.group(1)
                    section, i = parse_section(section_name, lines, i + 1)
                    structure[section_name] = section
            i += 1

        return structure

    def create_training_prompts(self, parsed_data: Dict) -> List[Dict]:
      prompts = []

      for example in parsed_data["examples"]:
          prompts.append({
              "system": "You are an expert in generating aero-f input files. Follow the examples closely.",
              "user": "Generate an aero-f input file based on this example:\n" + example,
              "assistant": example
          })

      for section, content in parsed_data["structure"].items():
          prompts.append({
              "system": "You are an expert in generating aero-f input files. Adhere to these structural guidelines.",
              "user": f"Generate an aero-f input file following these requirements:\n{content}",
              "assistant": ""
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
        # Enhanced prompt with detailed structure example and parameter guidance
        examples_text = ""
        for example in examples:
            examples_text += f"\n\nExample ({example['heading']}):\n"
            examples_text += self.format_structure_for_prompt(example["structure"])

        # Extract key information from user prompt
        simulation_type = "unknown"
        if "compressible flow simulation" in user_prompt.lower():
            simulation_type = "NavierStokes"  # Infer compressible flow implies Navier-Stokes
        elif "incompressible flow simulation" in user_prompt.lower():
            simulation_type = "Incompressible"  # Potential keyword for incompressible flow
        elif "steady" in user_prompt.lower():
            simulation_type = "Steady"
        elif "unsteady" in user_prompt.lower():
            simulation_type = "Unsteady"

        file_name = "unknown"
        match = re.search(r"(\w+\.msh)", user_prompt)
        if match:
            file_name = match.group(1)

        reynolds_number = "unknown"
        match = re.search(r"Re\s*(\d+)", user_prompt)
        if match:
            reynolds_number = match.group(1)

        mach_number = "unknown"
        match = re.search(r"Mach\s*([\d.]+)", user_prompt)
        if match:
            mach_number = match.group(1)

        accuracy_order = "unknown"
        if "second order" in user_prompt.lower():
            accuracy_order = "2"
        elif "first order" in user_prompt.lower():
            accuracy_order = "1"

        # Use knowledge base to infer parameters
        parameters = {}
        if simulation_type != "unknown":
            parameters["Problem"] = {"Type": simulation_type}

        if file_name != "unknown":
            parameters["Input"] = {"Geometry": file_name}

        if reynolds_number != "unknown":
            if "Input" not in parameters:
                parameters["Input"] = {}
            parameters["Input"]["ReynoldsNumber"] = reynolds_number  # Assuming you add this to your knowledge base

        if mach_number != "unknown":
            if "BoundaryConditions" not in parameters:
                parameters["BoundaryConditions"] = {"under Inlet": {}}
            parameters["BoundaryConditions"]["under Inlet"]["Mach"] = mach_number

        if accuracy_order != "unknown":
            if "Space" not in parameters:
                parameters["Space"] = {"under NavierStokes": {}}
            parameters["Space"]["under NavierStokes"]["Reconstruction"] = "Linear" if accuracy_order == "2" else "Constant"

            if "Time" not in parameters:
                parameters["Time"] = {"under Implicit": {}}
            parameters["Time"]["under Implicit"]["Order"] = accuracy_order
        # Construct parameter section of prompt
        parameter_section = self.format_structure_for_prompt(parameters)

        # Build the final prompt
        full_prompt = f"""You are an expert in generating aero-f input files.

        Here are some examples of aero-f input files, pay very close attention to the use of 'under' to create a hierarchical structure:

        {examples_text}

        The user has provided the following request:

        "{user_prompt}"

        Based on this request and the examples, generate an aero-f input file.

        Incorporate the following information extracted from the user prompt:

        - Simulation type: {simulation_type}
        - File name: {file_name}
        - Reynolds number: {reynolds_number}
        - Mach number: {mach_number}
        - Accuracy order: {accuracy_order}

        Use the following structure for the input file, filling in appropriate parameters based on the extracted information and the examples:

        {parameter_section}

        Make sure to use the 'under' keyword to create a structure like in the examples.
        """

        return self.query_llama(full_prompt)

    def format_structure_for_prompt(self, structure: Dict, indent_level=0) -> str:
      """Formats the tree-like structure into a string suitable for the prompt."""
      prompt_str = ""
      indent = "  " * indent_level

      for key, value in structure.items():
          if isinstance(value, dict):
              if key.startswith("under"):
                  prompt_str += f"{indent}{key} {{\n"
                  prompt_str += self.format_structure_for_prompt(value, indent_level + 1)
                  prompt_str += f"{indent}}}\n"
              else:
                  prompt_str += f"{indent}under {key} {{\n"
                  prompt_str += self.format_structure_for_prompt(value, indent_level + 1)
                  prompt_str += f"{indent}}}\n"
          else:
              prompt_str += f"{indent}{key} = {value};\n"

      return prompt_str

def generate_aero_f_input(user_prompt):
    processor = AeroFDocProcessor(
        base_url="https://frg.bitbucket.io/aero-f/",
        api_url="http://localhost:1234/v1",
        knowledge_base_file="knowledge_base.yaml"
    )

    html_content = processor.fetch_documentation()
    examples, structure = processor.parse_html(html_content)

    generated_file_content = processor.generate_input_file(user_prompt, examples)
    return generated_file_content

iface = gr.Interface(
    fn=generate_aero_f_input,
    inputs=[
        gr.Textbox(label="User Prompt", placeholder="Enter a detailed description of the desired simulation...")
    ],
    outputs=gr.Textbox(label="Generated Aero-F Input File", lines=10),
    title="Aero-F Input File Generator",
    description="Generate Aero-F input files using a local Llama model, with improved example extraction, structure awareness, and parameter knowledge."
)

iface.launch()