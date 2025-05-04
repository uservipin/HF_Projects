
'''
# Install dependencies
# !pip install -U langchain langsmith "unstructured[all-docs]" unstructured-client watermark

# Imports
import os
import json
import base64
import warnings
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import display
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.run_helpers import traceable  # LangSmith tracing
from langsmith.run_helpers import create_run_tree

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up LangSmith API
LANGSMITH_TRACING= True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANG_SMITH")
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "PDF_Image_QA_Tracking"


# Initialize LLM (Gemini Pro via LangChain)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

# Silence warnings
warnings.filterwarnings('ignore')


# LangSmith-tracked PDF parser
@traceable(name="PDF Partition")
def parse_pdf_with_unstructured(filename: str, output_dir: str):
    elements = partition_pdf(
        filename=filename,
        strategy='hi_res',
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_output_dir=output_dir,
    )
    return elements


# LangSmith-tracked function to convert image to base64
@traceable(name="Image to Base64")
def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# LangSmith-tracked Gemini image+text QA function
@traceable(name="Gemini LLM Image QA")
def ask_image_question(image_b64: str, prompt: str = "What do you see in this image?"):
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ])
    return llm.invoke([msg])


# 📄 Step 1: Parse PDF and extract elements
filename = "src/input/PDF/bain 1-10.pdf"
output_dir = "src/output/images_bain"
elements = parse_pdf_with_unstructured(filename, output_dir)

# Optional: Get all unique element types
element_dict = [el.to_dict() for el in elements]
unique_types = {item['type'] for item in element_dict}
print("Unique element types found:", unique_types)

# 📷 Step 2: Load image and convert to base64
image_path = "src/output/images_bain/figure-2-7.jpg"
image_b64 = encode_image_base64(image_path)

# 🧠 Step 3: Ask LLM a question about the image
response = ask_image_question(image_b64, prompt="What do you see in this image?")
print("Gemini Response:", response.content)
'''

# =======================================


# Imports
import os
import json
import base64
import warnings
from PIL import Image
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.run_helpers import traceable  # LangSmith tracing

# Load environment variables from .env
load_dotenv()

# ✅ Set up LangSmith Environment
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "PDF_Image_QA_Tracking"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# ✅ Set up Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

# Silence warnings
warnings.filterwarnings('ignore')


# ✅ LangSmith-tracked PDF parser
@traceable(name="PDF Partition")
def parse_pdf_with_unstructured(filename: str, output_dir: str):
    elements = partition_pdf(
        filename=filename,
        strategy='hi_res',
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_output_dir=output_dir,
    )
    return elements


# ✅ LangSmith-tracked function to convert image to base64
@traceable(name="Image to Base64")
def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ✅ LangSmith-tracked Gemini image+text QA function
@traceable(name="Gemini LLM Image QA")
def ask_image_question(image_b64: str, prompt: str = "What do you see in this image?"):
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ])
    return llm.invoke([msg])


# ✅ Run your steps normally (no create_run_tree)
filename = "src/input/PDF/bain 1-10.pdf"
output_dir = "src/output/images_bain"
elements = parse_pdf_with_unstructured(filename, output_dir)

element_dict = [el.to_dict() for el in elements]
unique_types = {item['type'] for item in element_dict}
print("Unique element types found:", unique_types)

image_path = "src/output/images_bain/figure-2-7.jpg"
image_b64 = encode_image_base64(image_path)

response = ask_image_question(image_b64, prompt="What do you see in this image?")
print("Gemini Response:", response.content)
