import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Tuple
import gradio as gr
from openai import OpenAI
import re

class AeroFDocProcessor:
    def __init__(self, base_url: str, api_url: str):
        self.base_url = base_url
        self.client = OpenAI(
            base_url=api_url,
            api_key="not-needed"
        )
        self.knowledge_base = self.create_knowledge_base()

    def create_knowledge_base(self) -> Dict:
      """Creates a basic knowledge base of Aero-F parameters."""
      knowledge_base = {
          "Problem": {
              "Type": {"description": "Type of problem (Steady, Unsteady, ...)", "values": ["Steady", "Unsteady"]},
              "Mode": {"description": "Mode of operation (Dimensional, NonDimensional)", "values": ["Dimensional", "NonDimensional"]},
          },
          "Input": {
              "Prefix": {"description": "Prefix for input files", "type": "string"},
              "Connectivity": {"description": "Connectivity file", "type": "string"},
              "Geometry": {"description": "Geometry file", "type": "string"},
              "Decomposition": {"description": "Decomposition file", "type": "string"},
              "CpuMap": {"description": "CPU map file", "type": "string"},
          },
          "Output": {
              "under Postpro": {
                  "Prefix": {"description": "Prefix for output files", "type": "string"},
                  "Residual": {"description": "Residual file", "type": "string"},
                  "Force": {"description": "Force file", "type": "string"},
                  "Mach": {"description": "Mach number file", "type": "string"},
                  "Frequency": {"description": "Output frequency", "type": "integer"},
              },
              "under Restart": {
                  "Prefix": {"description": "Prefix for restart files", "type": "string"},
                  "Solution": {"description": "Solution file", "type": "string"},
                  "RestartData": {"description": "Restart data file", "type": "string"},
                  "Frequency": {"description": "Restart frequency", "type": "integer"},
              },
          },
          "Equations": {
              "Type": {"description": "Type of equations (Euler, NavierStokes, ...)", "values": ["Euler", "NavierStokes", "Potential"]},
          },
          "BoundaryConditions": {
              "under Inlet": {
                  "Mach": {"description": "Mach number at inlet", "type": "float"},
                  "Alpha": {"description": "Angle of attack (degrees)", "type": "float"},
                  "Beta": {"description": "Sideslip angle (degrees)", "type": "float"},
              },
              "under Wall": {
                "Type": {"description": "Type of wall condition (NoSlip, etc...)", "values": ["NoSlip"]}
              },
              "under Outlet": {
                "Type": {"description": "Type of outlet condition (Pressure, etc...)", "values": ["Pressure"]},
                "Value": {"description": "Value for outlet condition", "type": "float"}
              }
              # Add more boundary condition types and parameters here
          },
          "Space": {
              "under NavierStokes": {
                  "Reconstruction": {"description": "Reconstruction method", "values": ["Linear", "Quadratic"]},
                  "Gradient": {"description": "Gradient calculation method", "values": ["Galerkin", "LeastSquares"]},
              },
          },
          "Time": {
              "MaxIts": {"description": "Maximum number of iterations", "type": "integer"},
              "Eps": {"description": "Convergence criterion", "type": "float"},
              "Cfl0": {"description": "Initial CFL number", "type": "float"},
              "CflMax": {"description": "Maximum CFL number", "type": "float"},
              "Ser": {"description": "Serialization parameter", "type": "float"},
              "under Implicit": {
                "MatrixVectorProduct": {"description": "Matrix-vector product method", "values": ["FiniteDifference"]},
                "under Newton": {
                    "MaxIts": {"description": "Maximum Newton iterations", "type": "integer"},
                    "under LinearSolver": {
                        "under NavierStokes": {
                            "Type": {"description": "Linear solver type", "values": ["Gmres"]},
                            "MaxIts": {"description": "Maximum linear solver iterations", "type": "integer"},
                            "KrylovVectors": {"description": "Number of Krylov vectors", "type": "integer"},
                            "Eps": {"description": "Linear solver tolerance", "type": "float"},
                            "Preconditioner.Type": {"description": "Preconditioner type", "values": ["Ras", "PointJacobi"]},
                        },
                    },
                },
              },
          },
          "Mesh": {
            "Path": {"description": "Path to mesh file", "type": "string"},
            "File": {"description": "Mesh file name", "type": "string"}
          },
          "Grid": {
            "under Zone": {
              "Name": {"description": "Name of the grid zone", "type": "string"},
              "Type": {"description": "Type of grid zone", "values": ["Structured", "Unstructured"]},
              "XStart": {"description": "X coordinate of the starting point", "type": "float"},
              "XEnd": {"description": "X coordinate of the ending point", "type": "float"},
              "YStart": {"description": "Y coordinate of the starting point", "type": "float"},
              "YEnd": {"description": "Y coordinate of the ending point", "type": "float"},
              "ZStart": {"description": "Z coordinate of the starting point", "type": "float"},
              "ZEnd": {"description": "Z coordinate of the ending point", "type": "float"}
            }
          },
          "Solver": {
            "under TimeStepping": {
              "Type": {"description": "Type of time stepping (Explicit, Implicit)", "values": ["Explicit", "Implicit"]},
              "Method": {"description": "Time stepping method (e.g., BDF, Runge-Kutta)", "values": ["BDF", "Runge-Kutta"]},
              "Order": {"description": "Order of the time stepping method", "type": "integer"}
            },
            "under Turbulence": {
              "Model": {"description": "Turbulence model (e.g., k-omega, Spalart-Allmaras)", "values": ["k-omega", "Spalart-Allmaras"]}
            }
          },
          "Output": {
            "under FileOutput": {
              "Frequency": {"description": "Output frequency for file output", "type": "integer"},
              "Fields": {"description": "Fields to be written in the output file", "type": "list of strings"}
            }
          }
          # Add more sections and parameters as needed
      }
      return knowledge_base

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
            examples_text += f"\n\nExample ({example['heading']}):\n{example['content']}"

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
            parameters["Input"]["ReynoldsNumber"] = reynolds_number

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
        parameter_section = ""
        for section, params in parameters.items():
            parameter_section += f"under {section} {{\n"
            for param, value in params.items():
                if isinstance(value, dict):
                    parameter_section += f"  under {param} {{\n"
                    for sub_param, sub_value in value.items():
                        parameter_section += f"    {sub_param} = {sub_value};\n"
                    parameter_section += "  }}\n"
                else:
                    parameter_section += f"  {param} = {value};\n"
            parameter_section += "}}\n"

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

        Make sure to use the 'under' keyword to create a structure like in the examples, for example:

        under Problem {{
          Type = Steady;
          Mode = NonDimensional;
        }}

        under Input {{
          Prefix = "data/";
          Connectivity = "wing.con";
          Geometry = "wing.msh";
          Decomposition = "wing.dec";
          CpuMap = "wing.4cpu";
        }}

        under Output {{
          under Postpro {{
            Prefix = "result/";
            Residual = "wing.res";
            Force = "wing.lift";
            Mach = "wing.mach";
            Frequency = 0;
          }}
          under Restart {{
            Prefix = "result/";
            Solution = "wing.sol";
            RestartData = "wing.rst";
            Frequency = 0;
          }}
        }}

        Equations.Type = Euler;

        under BoundaryConditions {{
          under Inlet {{
            Mach = 0.5;
            Alpha = 0.0;
            Beta = 0.0;
          }}
        }}

        under Space {{
          under NavierStokes {{
            Reconstruction = Linear;
            Gradient = Galerkin;
          }}
        }}

        under Time {{
          MaxIts = 10;
          Eps = 1.e-6;
          Cfl0 = 10.0;
          CflMax = 1.e99;
          Ser = 1.0;
          under Implicit {{
            MatrixVectorProduct = FiniteDifference;
            under Newton {{
              MaxIts = 1;
              under LinearSolver {{
                under NavierStokes {{
                  Type = Gmres;
                  MaxIts = 30;
                  KrylovVectors = 30;
                  Eps = 0.05;
                  Preconditioner.Type = Ras;
                }}
              }}
            }}
          }}
        }}

        Include relevant parameters for each section based on the type of simulation and the examples provided.
        """

        return self.query_llama(full_prompt)

def generate_aero_f_input(user_prompt):
    processor = AeroFDocProcessor(
        base_url="https://frg.bitbucket.io/aero-f/",
        api_url="http://localhost:1234/v1"
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