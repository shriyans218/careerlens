"""Extracts plain text from an uploaded resume file (pdf, docx, or txt)."""
import io
import pdfplumber
import docx


def read_resume(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    elif lower.endswith(".docx"):
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)
    else:
        # plain text fallback
        return file_bytes.decode("utf-8", errors="ignore")
