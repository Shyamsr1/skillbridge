import json
from langdetect import detect
from transformers import pipeline
import torch
from app import resume_parser_dl as parser

# Load DL language detection pipeline
lang_pipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Deep Learning-based language detection using pipeline (no tokenizers)
def detect_language_dl(text):
    try:
        result = lang_pipeline(text, truncation=True)[0]
        return result['label']
    except Exception as e:
        print("DL Language detection error:", e)
        return "unknown"

# Choose detection method: 'langdetect' or 'dl'
def detect_resume_language(resume_path, method="langdetect"):
    try:
        text = parser.extract_text_from_pdf(resume_path)
        if method == "dl":
            return detect_language_dl(text)
        else:
            return detect(text)
    except Exception as e:
        print("Language detection error:", e)
        return "unknown"

def parse_resume_with_language(resume_path, skill_list=None, method="langdetect"):
    parsed_data = parser.parse_resume(resume_path)
    parsed_data["DetectedLanguage"] = detect_resume_language(resume_path, method=method)
    return parsed_data
