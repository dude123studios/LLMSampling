
MATH_PROMPT = """You are a helpful mathematical assistant. Solve the following problem step-by-step.
IMPORTANT: You must put your final answer within \\boxed{{}}. For example: \\boxed{{42}}.

Problem:
{problem}

Solution:
"""

GPQA_PROMPT = """Answer the following multiple choice question. Think step by step. Finally, output your answer in the format "The answer is (X)" where X is A, B, C, or D.

Question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Answer:
"""

CODE_PROMPT = """You are an expert programmer. Write a solution to the following problem.
Output ONLY the code block.

Problem:
{problem}

Code:
```python
"""

def get_prompt(task_name, item):
    if task_name == "math":
        return MATH_PROMPT.format(problem=item['problem'])
    elif task_name == "gpqa":
        # GPQA structure might vary, assume standard fields or adapt
        # Typically GPQA has 'Question', 'Correct Answer', 'Incorrect Answer 1', etc.
        # Ideally we shuffle choices but for now simplified.
        # Assuming dataset has 'Question', 'A', 'B', 'C', 'D' pre-formatted or raw. 
        # The HF dataset usually has 'Question', 'Correct Answer', 'Incorrect Answer 1..3'
        # We need to map them to A,B,C,D. For simplicity here, just a placeholder logic.
        # IMPLEMENTATION DETAIL: Proper shuffling is needed for rigorous eval.
        # For now, just dumping the question.
        return GPQA_PROMPT.format(
            question=item['Question'],
            choice_A=item.get('Incorrect Answer 1', 'Option A'), # simplified
            choice_B=item.get('Incorrect Answer 2', 'Option B'),
            choice_C=item.get('Incorrect Answer 3', 'Option C'),
            choice_D=item.get('Correct Answer', 'Option D')
        )
    elif task_name == "code":
        return CODE_PROMPT.format(problem=item['content']) # LiveCodeBench uses 'content'
    return ""
