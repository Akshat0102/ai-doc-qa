import os
from PyPDF2 import PdfReader
import docx


def load_text(file_path: str) -> str:

    """Function to load text from file"""

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:

    """Function to extract text from PDF files using PyPDF2 library"""

    reader = PdfReader(file_path)
    text = []

    for page in reader.pages:
        extracted = page.extract_text()
        if (extracted):
            text.append(extracted)

    return "\n".join(text)


def load_docx(file_path: str) -> str:

    """Function to extract text from .docx file"""

    doc = docx.Document(file_path)
    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return "\n".join(text)


def load_file(file_path: str) -> str:
   
    """Function to load file based on its extension"""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)

    elif ext == ".pdf":
        return load_pdf(file_path)

    elif ext == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def clean_text(text: str) -> str:

    """Final Function to clean the loaded text from whitespaces and line breaks"""

    return (
        text.replace("\r", " ")
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
    )