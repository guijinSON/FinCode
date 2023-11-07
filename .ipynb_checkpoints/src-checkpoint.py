import openai
import pandas as pd
openai.api_key = "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"


def get_answer(instance):
    completion = openai.ChatCompletion.create(
                          model="gpt-3.5-turbo-1106",
                          messages=[
                            {"role": "user", "content": instance}
                          ],
                          temperature=0.7,
                          n=1
                        )
    output = completion.choices[0].message["content"]
    return output

def get_answer_gpt4(instance):
    completion = openai.ChatCompletion.create(
                          model="gpt-4-0613",
                          messages=[
                            {"role": "user", "content": instance}
                          ],
                          stop= "### Task:",
                          temperature=0.7,
                          n=1
                        )
    output = completion.choices[0].message["content"]
    return output

def gpt4_eval(instance):
    completion = openai.ChatCompletion.create(
                          model="gpt-4-0613",
                          messages=[
                            {"role": "user", "content": instance}
                          ],
                          stop= "### Task:",
                          temperature=0.2,
                          n=1
                        )
    output = completion.choices[0].message["content"]
    return output


def rewrite_course_description(course_title, course_description):
    template ="""You will be given a course title and its description. Extend the course description and make it more technically descriptive. Only discuss the core academic concepts for the course, not administrative matters like sexam schedules, prerequisites, projects, explanations of difficulty levels, homework, graduate conditions, textbook references and etc.

### Course Title: {course title}
### Course Description: {course description}
### Extended Course Description:"""
    
    template = template.replace("{course title}",course_title)
    template = template.replace("{course description}",course_description)
    output = get_answer(template)
    return output


def get_core_concepts(category, input_list):
    query = f"""You will be given a category and list of course titles and descriptions that fit in the category.
I am trying to compose a list of research papers for the category. 
To do so I need a list of core academic concepts that appear in the category to use as search queries.
Make a detailed list of core academic concepts that can be used to search research papers of the category.
Write in a numbered list style. 

- Only academic concepts in factuality, knowledge dimensions like theories and cases.
- List concepts in order of importance.
- Make sure the concept falls in the category.

### Category: {category}
### List of Course titles and descriptions: {input_list}
### Core Academic Concepts"""
    output = get_answer(query)
    return output

def get_question(category, concept, abstract):
    template = """Objective: The purpose is to design two-step questions that test (1) ability to write appropriate code based on financial concepts and (2) ability to interpret the result of  the first step in a statistical and financial manner. This two-fold approach aims to evaluate the ability of models to work with humans for novel discovery in finance.

Instructions:

(1) You will be given a category, concept, and an abstract of a research paper from the category and concept. 

(2) Read the abstract, write a single line summary on how the paper differs from other works.

(3) Referring the abstract, its data, experimental setup, and results formulate a coding question. 

(4) Formulate a subsequent reasoning question. This question must be connected with analyzing the results of the former coding question.

(5) If you question requires an input data, preset hyper parameter, or etc illustrate them. You don't have to make sample data just explain them.

Constraints:

(1) The coding question should ask for quantifiable result which can be statistically or financially interpreted, instead of a model, class, or function.  You must specify a specific model to use, you can't simply ask to compare.

(2) The coding questions must be a simple single-step question.

(3) The reasoning question must involve statistical analysis. For instance, evaluating statistical significance, comparing metrics, assessing model fits, providing financial insights, or etc.

(4) You cannot ask for additional coding or analysis in the reasoning question. If you are aiming for an analysis you should run all the code in the coding question.

(5) Difficulty for Coding Question: [Easy: Running statistical analysis, simple numerical computation etc]

(6) Return in a dictionary in the following format: \{summary:  "...", input_data: "..." Coding_Q: "...", Reasoning_Q: "..."\}

(7) Do not have generate answer for the questions.

Provided Information:

### category: {category}
### concept: {concept}
### abstract: {abstract}

### return_dictionary:"""
    template = template.replace("{category}",category)
    template = template.replace("{concept}",concept)
    template = template.replace("{abstract}",abstract)
    return template


def get_refine_query(input_data,coding_question,reasoning_question):
    template = """Objective: The purpose is to design two-step questions that test (1) ability to write appropriate code based on financial concepts and (2) ability to interpret the result of  the first step in a statistical and financial manner. This two-fold approach aims to evaluate the ability of models to work with humans for novel discovery in finance.
    
#### Instructions:

(1) You will be given <input_data>, <coding_question>, and <reasoning_question>.

(2) Refine the <coding_question> so to make sure it is consisted of a single-step coding task that will always return an identical output. The output will be evaluated via unit test.

(3) Return how the output should look like in detail. (ex. an interger, an array etc)

(3) Refine the <reasoning_question> so to make sure it leverage the output of the <coding_question> to generate meaningful insight. The <reasoning_question> should not require any additional calculation other than the output given from the <coding_question>.

(4) Specify the <input_data>. Explain in detail the columns and data required, the timeframe and etc needed to collect it.

### Constraints:

(1) Return in a dictionary in the following format: \{Coding_Q: ""..."", Reasoning_Q: ""..."",input_data: ""..."", output_data: ""...""\}

(2) Do not have generate answer for the questions.

### <input_data>: 
{input_data}

### <coding_question>: 
{coding_question}

### <reasoning_question>: 
{reasoning_question}

### return dictionary:"""
    template = template.replace("{input_data}",input_data)
    template = template.replace("{coding_question}",coding_question)
    template = template.replace("{reasoning_question}",reasoning_question)
    return template


def construct_coding_question(coding_q,dataset_card,duid):
    df = pd.read_csv(f"data/{duid}.csv")
    query = f"""You will be given (1) a coding question, (2) a dataest card that explains the provided dataset, (3) the first five row of the dataset in latex format, and (4) dataset path. Solve the coding question and return your answer in python. 
    
### Coding Question:
{coding_q} Print the output.

### Dataset Card:
{dataset_card}

### Dataset Head:
{df.head().to_latex()}

### Dataset Path: data/uid.csv

### Answer:"""
    return query

def construct_reasoning_question(reasoning_q, coding_q_output):
    query = f"""You will be given (1) a reasoning questioon, (2) an output from a code. Refer to the analysis results provided in (2) to solve (1)
    
### Reasoning Question:
{reasoning_q}

### Output from Coding Question: 
{coding_q_output}

### Answer:"""
    return query

def construct_eval_query(reasoning_q, reasoning_q_output):
    query = f"""Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user question displayed below. Your evaluation should consider factors
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
the response. Begin your evaluation by providing a short explanation. Be as objective as
possible. After providing your explanation, please rate the response on a scale of 1 to 10
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{reasoning_q}
[The Start of Assistant’s Answer]
{reasoning_q_output}
[The End of Assistant’s Answer]"""
    return query