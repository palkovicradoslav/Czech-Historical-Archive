import re
import logging
import json
import ast
import os
from dotenv import load_dotenv

load_dotenv()


def setup_logger(verbosity=True):
    logging.basicConfig(
        level=logging.INFO if verbosity else logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_words(sentence):
    """Splits sentence into words"""
    words = sentence.split()
    return [word.strip(".,;:!?()[]{}\"'") for word in words]


def fix_json_output(json_str):
    """Clean a json string"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def fix_corrected_text_line(line):
    """Potentially fixes text line formatting before parsing"""
    if '"corrected_text":' in line:
        colon_index = line.find(':')
        first_quote = line.find('"', colon_index)
        if first_quote == -1:
            return line

        start_val = first_quote + 1
        end_val = line.rfind('"')
        if end_val <= start_val:
            return line

        value = line[start_val:end_val]
        fixed_value = value.replace('"', r'\"')
        fixed_line = line[:start_val] + fixed_value + line[end_val:]
        return fixed_line
    return line


def correct_llm_output(response):
    """Gets JSON output from LLM response, tries to fix common issues"""
    if response.status_code == 200:
        result = response.json()

        result = result.get("choices", [{}])[0].get(
            "message", {}).get("content", "")

        result = fix_json_output(result)

        fixed_lines = [fix_corrected_text_line(
            line) for line in result.splitlines()]
        fixed_json_str = "\n".join(fixed_lines)
        try:
            corrected_json = json.loads(fixed_json_str)

            return corrected_json
        except Exception:
            logging.warning("Failed to parse JSON output. Trying ast")
            try:
                return ast.literal_eval(result)
            except Exception:
                logging.error("Loading failed completely:\n" + result)
    else:
        logging.error("Error: " + response.status_code + response.text)


def get_api_keys():
    """Return API keys from environment variables"""
    api_keys = {}

    openrouter = os.getenv('OPENROUTER_API_KEY')
    gemini = os.getenv('GEMINI_API_KEY')

    if openrouter:
        api_keys['OPENROUTER_API_KEY'] = openrouter
    if gemini:
        api_keys['GEMINI_API_KEY'] = gemini

    if not api_keys:
        print('No API keys found in the environment')

    return api_keys
