import contextlib
import io
import sys
from src import construct_coding_question, construct_reasoning_question, construct_eval_query, gpt4_eval


@contextlib.contextmanager
def capture_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def run_python_code(code_str):
    # Capture the output during code execution
    with capture_output() as (out, err):
        # Execute the compiled code object
        try:
            # Execute the code in the global namespace
            exec(code_str, globals())
        except Exception as e:
            # Return an error message if execution fails
            print(str(e), file=err)

    # Return a dictionary with standard output and any error messages
    return {'output': out.getvalue(), 'error': err.getvalue()}

# Function to extract a Python code snippet from a string that uses triple backticks as used in Markdown
def extract_code_from_string(input_string):
    # Split the string by lines
    lines = input_string.split('\n')
    
    # Initialize an empty list to hold the code lines
    code_lines = []
    
    # Flag to indicate whether we are inside a code block
    inside_code_block = False

    # Iterate over each line in the input string
    for line in lines:
        # Check if the line contains the start or end of a code block
        if line.strip().startswith('```python'):
            inside_code_block = True
            continue
        elif line.strip() == '```' and inside_code_block:
            inside_code_block = False
            continue
        
        # If we are inside a code block, add the line to the code lines
        if inside_code_block:
            code_lines.append(line)
    
    # Join the code lines into a single string
    code_string = '\n'.join(code_lines)
    
    return code_string

def solve_eval(row,get_answer_fn):
    coding_q, reasoning_q, dataset_card, duid = row["Coding Question"], row["Reasoning Question"], row["Dataset Card"], row["duid"]
    coding_q_query = construct_coding_question(coding_q, dataset_card, duid)
    coding_q_generated = get_answer_fn(coding_q_query)
    code = extract_code_from_string(coding_q_generated)
    code = code.replace("data/uid.csv",f"data/{duid}.csv")
    coding_q_output = run_python_code(code)

    if len(coding_q_output["output"]) > 1:
        reasoning_q_query = construct_reasoning_question(reasoning_q,coding_q_output["output"])
        reasoning_q_generated = get_answer_fn(reasoning_q_query)
        eval_query = construct_eval_query(reasoning_q, reasoning_q_generated)   
        eval_generated = gpt4_eval(eval_query)
        reasoning_score = eval(eval_generated.split("Rating:")[-1])[0][0]
    else: 
        reasoning_q_query = "Coding Failure"
        reasoning_q_generated = "Coding Failure"
        eval_query = "Coding Failure"
        eval_generated = "Coding Failure"
        reasoning_score = 0
        
    row_dict = {
            "coding_q": coding_q, 
            "reasoning_q":reasoning_q, 
            "dataset_card":dataset_card,
            "duid":duid,
            "coding_q_query":coding_q_query,
            "coding_q_generated":coding_q_generated,
            "codeing_q_output":coding_q_output,
            "reasoning_q_query":reasoning_q_query,
            "reasoning_q_generated":reasoning_q_generated,
            "eval_query":eval_query,
            "eval_generated":eval_generated,
            "reasoning_score":reasoning_score
    }
    return row_dict