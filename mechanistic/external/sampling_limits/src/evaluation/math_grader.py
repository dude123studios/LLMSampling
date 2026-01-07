import re

def extract_boxed(text):
    if not text: return None
    
    # 1. Find all `\boxed` indices
    # We want the LAST valid boxed content
    matches = [m.start() for m in re.finditer(r"\\boxed", text)]
    if not matches:
        return None
        
    for idx in reversed(matches):
        # Scan forward for the opening brace, allowing whitespace
        # text[idx] is '\', text[idx+1:idx+6] is 'boxed'
        scan_idx = idx + 6 
        while scan_idx < len(text) and text[scan_idx].isspace():
            scan_idx += 1
            
        if scan_idx >= len(text) or text[scan_idx] != "{":
            continue # Malformed or missing brace
            
        # Brace balancing
        balance = 1
        content = ""
        scan_idx += 1 # Enter content
        
        valid_extraction = False
        while scan_idx < len(text):
            char = text[scan_idx]
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
                
            if balance == 0:
                valid_extraction = True
                break
            
            content += char
            scan_idx += 1
            
        if valid_extraction:
            return content.strip()
            
    return None

def extract_last_number(text):
    # Regex for last number
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return matches[-1]
    return None

def normalize_answer(s):
    if s is None:
        return ""
    # Simple normalization: specific for 1/2 vs 0.5 etc.
    # In a full impl this uses SymPy
    s = s.strip()
    try:
        if "/" in s:
            num, den = s.split("/")
            return str(float(num) / float(den))
    except:
        pass
    return s

def grade_math(prediction, reference):
    pred = extract_boxed(prediction)
    if pred is None:
        pred = extract_last_number(prediction)
    
    norm_pred = normalize_answer(pred)
    norm_ref = normalize_answer(extract_boxed(reference)) # Ref is often boxed
    
    return norm_pred == norm_ref
