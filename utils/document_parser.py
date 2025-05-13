import fitz  # PyMuPDF
from docx import Document
import io

# This file will contain functions for parsing different document types.

def parse_pdf(file_content: bytes) -> str:
    """Parses PDF file content and extracts text."""
    text = ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""
    return text

def parse_docx(file_content: bytes) -> str:
    """Parses DOCX file content and extracts text."""
    text = ""
    try:
        doc = Document(io.BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        return ""
    return text

def parse_txt(file_content: bytes) -> str:
    """Parses TXT file content and extracts text."""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1') # Fallback encoding
        except Exception as e:
            print(f"Error parsing TXT: {e}")
            return ""
    except Exception as e:
        print(f"Error parsing TXT: {e}")
        return ""