import re

def grade_gpqa(prediction, reference):
    # Reference is typically 'A', 'B', 'C', or 'D'
    
    # Look for "The answer is (X)"
    match = re.search(r"The answer is \(([A-D])\)", prediction)
    if match:
        pred_choice = match.group(1)
        return pred_choice == reference
        
    # Fallback: look for last (X)
    matches = re.findall(r"\(([A-D])\)", prediction)
    if matches:
        return matches[-1] == reference
        
    return False
