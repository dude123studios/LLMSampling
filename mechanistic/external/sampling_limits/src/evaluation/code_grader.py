
def extract_code_block(text):
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text

def prepare_code_eval(prediction, reference, item_id):
    """
    Returns a dict object ready for JSONL dumping.
    Does NOT execute code.
    """
    code = extract_code_block(prediction)
    return {
        "task_id": item_id,
        "generation": code,
        "prompt": reference # LiveCodeBench might need specific format
    }
