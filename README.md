# Aero-F Input File Generator using Local Llama Model

This project implements a system for generating Aero-F input files using a local Llama language model, specifically leveraging Meta's Llama 3 via LM Studio's local inference server. The script fetches and parses the Aero-F documentation, extracts examples and structural information, and then utilizes the Llama model to generate input files based on user prompts.

## Overview

The `llm-aero-v1.py` script is designed to simplify the creation of input files for the Aero-F computational fluid dynamics (CFD) solver. It automates the process by intelligently interpreting user requests and generating corresponding configuration files. The script performs the following key functions:

1. **Fetches Aero-F Documentation:** Retrieves the HTML documentation from the official Aero-F website.
2. **Parses HTML:** Extracts relevant information, including example input files and structural guidelines, from the HTML content.
3. **Knowledge Base Creation:** Establishes a basic knowledge base of Aero-F parameters, their descriptions, and possible values.
4. **Prompt Engineering:** Constructs detailed prompts for the Llama model, incorporating examples, structural guidelines, and user-specified parameters.
5. **Llama Model Interaction:** Queries the locally running Llama model (via LM Studio's API) to generate the input file content.
6. **User Interface:** Provides a simple Gradio-based web interface for users to input their simulation requirements and receive the generated input file.

## Dependencies

Before running the script, ensure you have the following dependencies installed:

1. **Python:** The script is written in Python and requires Python 3.9 or higher.
2. **LM Studio:** This application is used to run the Llama model locally and provides the API endpoint for interaction. Download and install it from the [official website](https://lmstudio.ai/).
3. **Required Python Libraries:**

    *   `requests`: For making HTTP requests to fetch the Aero-F documentation and interact with the LM Studio API.
    *   `beautifulsoup4`: For parsing HTML content.
    *   `gradio`: For creating the web-based user interface.
    *   `openai`: For interacting with the OpenAI-compatible API provided by LM Studio.
    *   `typing`: For type hinting.
    *   `re`: For regular expression.

    Install these libraries using pip:

    ```bash
    pip install requests beautifulsoup4 gradio openai typing re
    ```

## Setup and Usage

1. **Install LM Studio:** Download, install, and launch LM Studio.
2. **Download and Load the Llama Model:**
    *   In LM Studio, go to the "Model" tab.
    *   Search for and download a suitable Llama model (e.g., "Meta-Llama-3-8B-Instruct").
    *   Once downloaded, select the model to load it.
3. **Start the Local Inference Server:**
    *   In LM Studio, navigate to the "Local Server" tab.
    *   Click "Start Server" to launch the local inference server.
    *   Note the API endpoint displayed (e.g., `http://localhost:1234/v1`).
4. **Clone the Repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
5. **Run the Script:**
    ```bash
    python llm-aero-v1.py
    ```
6. **Access the Web Interface:**
    *   The script will launch a Gradio web interface. Open your web browser and go to the URL provided in the console (usually `http://127.0.0.1:7860`).
7. **Enter User Prompt:**
    *   In the web interface, enter a detailed description of the desired Aero-F simulation in the "User Prompt" textbox.
    *   Be as specific as possible, including details like the type of simulation (steady/unsteady, compressible/incompressible), file names, Reynolds number, Mach number, accuracy order, and any other relevant parameters.
8. **Generate Input File:**
    *   Click the "Submit" button. The script will process your request, query the Llama model, and display the generated Aero-F input file in the "Generated Aero-F Input File" textbox.

## Code Explanation

### `AeroFDocProcessor` Class

This class encapsulates the core functionality of the script.

#### `__init__(self, base_url: str, api_url: str)`

*   Initializes the `AeroFDocProcessor` with the base URL of the Aero-F documentation and the API URL of the local Llama model.
*   Creates an `OpenAI` client to interact with the LM Studio API.
*   Calls `create_knowledge_base()` to initialize a dictionary containing basic knowledge about Aero-F parameters.

#### `create_knowledge_base(self) -> Dict`

*   Defines a dictionary representing a basic knowledge base of Aero-F parameters.
*   Includes sections like "Problem", "Input", "Output", "Equations", "BoundaryConditions", "Space", "Time", "Mesh", "Grid", and "Solver".
*   Each section may contain parameters with descriptions and possible values.

#### `fetch_documentation(self) -> str`

*   Fetches the HTML content from the Aero-F documentation website using the `requests` library.
*   Returns the HTML content as a string.

#### `parse_html(self, html_content: str) -> Tuple[List[Dict], Dict]`

*   Parses the HTML content using `BeautifulSoup`.
*   Extracts examples of Aero-F input files from the "Examples" section of the documentation.
*   Identifies headings (h4 to h9) preceding each example block and stores them along with the example content.
*   Returns a tuple containing a list of example dictionaries (each with "heading" and "content") and an empty dictionary for structure (which is currently unused but could be extended to extract structural information).

#### `create_training_prompts(self, parsed_data: Dict) -> List[Dict]`

*   (Currently not used for training but could be adapted for fine-tuning a model).
*   Generates prompts for potential fine-tuning of the Llama model.
*   Creates prompts based on the extracted examples and (in the future) structural guidelines.

#### `query_llama(self, prompt: str) -> str`

*   Sends a prompt to the local Llama model via the LM Studio API.
*   Uses the `OpenAI` client to create a chat completion.
*   Specifies the model name ("local-model"), messages (system and user prompts), temperature, and maximum tokens.
*   Returns the generated response from the model or an error message if something goes wrong.

#### `generate_input_file(self, user_prompt: str, examples: List[Dict]) -> str`

*   This is the main function that generates the Aero-F input file based on the user prompt and extracted examples.
*   It first creates a detailed prompt for the Llama model, including:
    *   A system message indicating the role of the model ("expert in generating aero-f input files").
    *   A concatenation of the extracted examples.
    *   The user's input prompt.
    *   Extracted key information from the user prompt (simulation type, file name, Reynolds number, Mach number, accuracy order) using regular expressions and simple string matching.
    *   A parameter section based on the extracted information and the knowledge base.
    *   Explicit instructions on how to use the `under` keyword to create the hierarchical structure of the input file, with illustrative examples.
*   It then calls `query_llama()` to send the constructed prompt to the Llama model and receive the generated input file content.

### `generate_aero_f_input(user_prompt)`

*   This function is the entry point for the Gradio interface.
*   It creates an instance of the `AeroFDocProcessor`.
*   Fetches and parses the Aero-F documentation.
*   Calls `generate_input_file()` to generate the input file content based on the user prompt.
*   Returns the generated content to be displayed in the Gradio interface.

### Gradio Interface

*   The `gr.Interface` creates a simple web interface with:
    *   An input textbox for the user to enter their prompt.
    *   An output textbox to display the generated Aero-F input file.
    *   A title and description for the application.

## Future Improvements

1. **Enhanced HTML Parsing:** Improve the `parse_html` function to extract more structural information from the documentation, such as parameter dependencies and constraints.
2. **Fine-tuning:** Adapt the `create_training_prompts` function and potentially fine-tune the Llama model on a dataset of Aero-F input files to improve its accuracy and understanding of the specific syntax and structure.
3. **Error Handling and Validation:** Implement more robust error handling and validation of the generated input files to ensure they are syntactically correct and conform to Aero-F's requirements.
4. **Parameter Inference:** Enhance the parameter inference logic in `generate_input_file` to more accurately deduce parameter values from the user prompt based on the knowledge base and context.
5. **Interactive Interface:** Develop a more interactive Gradio interface that allows users to select parameters from dropdown menus, checkboxes, or other input elements, potentially reducing the reliance on free-form text prompts.
6. **Code Refactoring:** Break down the `generate_input_file` into smaller, more modular functions to improve readability and maintainability.
7. **Unit Testing:** Add unit tests to ensure the code functions as expected and to prevent regressions as new features are added.

## Disclaimer

This project is intended for educational and research purposes. The generated Aero-F input files should be carefully reviewed and validated before being used in actual simulations. The authors of this project are not responsible for any errors or issues that may arise from the use of the generated files.

By following these instructions and understanding the code structure, you can effectively use the `llm-aero-v1.py` script to generate Aero-F input files using a locally running Llama model. Remember to consult the Aero-F documentation and validate the generated files before using them in your simulations.
