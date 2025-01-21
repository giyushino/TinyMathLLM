from sympy import * 
import numpy as np 
import random 
import json 

question_types = ["Addition", "Multiplication", "Derivation", "Integration", "Matrix Multiplication", "Determinant"]
calculus_questions = ["ln","power", "fraction", "trig"]
NUM_QUESTIONS = 100000


def addition():
    num1 = random.randint(0, 10000)
    num2 = random.randint(0, 10000)
    solution = num1 + num2
    
    return num1, num2, solution

def multiplication():
    num1 = random.randint(0, 10000)
    num2 = random.randint(0, 10000)
    solution = num1 * num2

    return num1, num2, solution

def derivation():
    question = random.choice(calculus_questions)
    x = symbols("x")
    if question == "trig":
        equation = random.choice(["sin","cos", "tan", "csc", "sec", "cot"]) + "(x)"
    elif question == "power":
        equation = str(random.randint(1, 10)) + "*x**" + str(random.randint(1, 10))
    elif question == "fraction":
        equation = "1/x**" + str(random.randint(1, 10))
    else:
        equation = "ln(x)"

    return equation, diff(equation, x)


def integration(): 
    question = random.choice(calculus_questions)
    x = symbols("x")
    if question == "trig":
        equation = random.choice(["sin","cos", "tan", "csc", "sec", "cot"]) + "(x)"
    elif question == "power":
        equation = str(random.randint(1, 10)) + "*x**" + str(random.randint(1, 10))
    elif question == "fraction":
        equation = "1/x**" + str(random.randint(1, 10))
    else:
        equation = "ln(x)"

    return equation, integrate(equation, x)

def matrix_mult(low = 1, high = 20, row = random.randint(2, 5), column = random.randint(2, 7)):
    m1 = np.random.randint(low, high, size = (row, column))
    m2 = np.random.randint(low, high, size = (column, row))

    return m1, m2, np.matmul(m1, m2)

def matrix_det(low = 1, high = 20, dim = random.randint(2, 5)):
    m1 = np.random.randint(low, high, size = (dim, dim))
    
    return m1, np.linalg.det(m1)


def create_question_answer():
    """
    Creates dataset with sample question, the chatbot is supposed to extract the important information + determine what type of problem it is
    
    Args:
        None
    Returns:
        str: Random input and expected output

    """
    question = random.choice(question_types)
    prompt = {}
    
    if question == "Addition":
        num1, num2, solution = addition()
        user_question = f"What is {num1} + {num2}?"
        llm_answer = f"{num1} + {num2} = {solution}"
    elif question == "Multiplication":
        num1, num2, solution = multiplication()
        user_question = f"What is {num1} x {num2}?"
        llm_answer = f"{num1} x {num2} = {solution}"
    elif question == "Derivation":
        equation, diff = derivation()   
        user_question = f"What is the derivative of {equation}"
        llm_answer = f"The derivative of {equation} is {diff}"
    elif question == "Integration":
        equation, integral = integration()
        user_question = f"What is the integral of {equation}"
        llm_answer = f"The integral of {equation} is {integral}"
    elif question == "Matrix Multiplication":
        m1, m2, solution = matrix_mult()
        user_question = f"What is {m1} times {m2}"
        llm_answer = f"{m1} times {m2} = {solution}"
    else: 
        m1, det = matrix_det()
        user_question = f"What is the determinant of {m1}"
        llm_answer = f"The determinant of {m1} is {det}"

    prompt["messages"] = [
            {"role": "system", "content": "You are a mathematics professor who helps students with their math problems. Provide them with the answer."},
            {"role": "user", "content": f"{user_question}"},
            {"role": "assistant", "content": f"{llm_answer}."}
            ]

    return prompt


def create_question_extract():

    """
    Creates dataset with sample question, the chatbot is supposed to answer the math problem
   
    Args:
        None
    Returns:
        str: Random input and expected output
   """

    question = random.choice(question_types)
    prompt = {}
    
    if question == "Addition":
        num1, num2, solution = addition()
        user_question = f"What is {num1} + {num2}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Addition. Important values: {num1}, {num2}"
    elif question == "Multiplication":
        num1, num2, solution = multiplication()
        user_question = f"What is {num1} x {num2}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Multiplication. Important values: {num1}, {num2}"
    elif question == "Derivation":
        equation, diff = derivation()   
        user_question = f"What is the derivative of {equation}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Derivation. Important values: {equation}"
    elif question == "Integration":
        equation, integral = integration()
        user_question = f"What is the integral of {equation}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Integration. Important values: {equation}"
    elif question == "Matrix Multiplication":
        m1, m2, solution = matrix_mult()
        user_question = f"What is {m1} times {m2}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Matrix Multiplication. Important values: {m1}, {m2}"
    else: 
        m1, det = matrix_det()
        user_question = f"What is the determinant of {m1}? What kind of question is this, and what are the important values?"
        llm_answer = f"Question type: Matrix Determinant. Important values: {m1}"

    prompt["messages"] = [
            {"role": "system", "content": "You are a mathematics professor who helps students with their math problems. Provide them with the answer."},
            {"role": "user", "content": f"{user_question}"},
            {"role": "assistant", "content": f"{llm_answer}."}
            ]

    return prompt

def create_dataset(dataset_type, num_questions = NUM_QUESTIONS):
    if dataset_type == "extract":
        with open(r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\datasets\testset\extract.json", "w") as file:
            for i in range(num_questions):
                line = create_question_extract()
                file.write(json.dumps(line) + "\n")
                print(f"Line {i} created")
    elif dataset_type == "answer":
        with open(r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\datasets\testset\answer.json", "w") as file:
            for i in range(num_questions):
                line = create_question_answer()
                file.write(json.dumps(line) + "\n")
                print(f"Line {i} created")
    else:
        print("Invalid dataset_type")


create_dataset("extract", 1000) 
create_dataset("answer", 1000)
