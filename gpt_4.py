#%%
import os
import openai
import pandas as pd
import numpy as np
import random
import re
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
#from dotenv import load_dotenv

# Load environment 
#load_dotenv()
gpt4_key = os.environ.get('GPT_key')
openai.api_key = gpt4_key

# Helper Functions
# Tenacity.retry decorator
# To add random exponential backoff to requests so we don't hit the rate limit
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Get GPT response while predefining some parameters
# I chose davinci-003 because it offers a short response, 
# after which I can ask it to explain itself if required.
def get_gpt_response(prompt_text, model_name = "gpt-4", max_tokens_no = 1000):
    response = completion_with_backoff(
        model = model_name, 
        prompt = prompt_text, 
        temperature = 0.5,
        max_tokens = max_tokens_no
        )
    
    return response.choices[0].text.strip()

def get_chat_response(prompt_text):
    # Messages sample:
    # {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "user", "content": "Who won the world series in 2020?"},
    # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    # {"role": "user", "content": "Where was it played?"}
    # System predefines the chatbot role
    # Then user messages are from the user
    # and Assistant messages are replies from the chatbot

    response = chatcompletion_with_backoff(
        model="gpt-4",
        messages=[
            {'role': "system", 'content': 'You are a radiologist taking an exam. Please choose one choice only, giving the number of the option.'},
            {'role': 'user', 'content': prompt_text}
        ], 
        temperature = 0.5
    )
    return response['choices'][0]['message']['content']

def get_chat_response_3(first_prompt_text, response_1, second_prompt_text):
    response = chatcompletion_with_backoff(
        model="gpt-4",
        messages=[
            {'role': "system", 'content': 'You are a radiologist taking an exam. Please choose one choice only, giving the number of the option.'},
            {'role': 'user', 'content': first_prompt_text},
            {'role': 'assistant', 'content': response_1},
            {'role': 'user', 'content': second_prompt_text}
        ], 
        temperature = 0.5
    )
    return response['choices'][0]['message']['content']

# Each qn needs to be a class...
# Inputs: 
# test_qn.qn() returns the question, formatted
# test_qn.a-e should return "A. <OPTION>"
# test_qn.ans should return "C. <OPTION>"
# test_qn.ans_index should return "C"
class test_qn:
    """Each question needs to be a class
    Inputs:
        Various parts of the df:
        columns 'Qn_Text', 'A'-'E', 'Ans'

    Returns: Nil

    Relevant properties:
        test_qn.qn              Question and options all in 1 line
        test_qn.ans             Correct answer with the option
        test_qn.ans_index       Correct answer choice (0-4)
        test_qn.gpt_response    AI answer choice - ?May be str or int?
        test_qn.gpt_explanation AI explanation for response given
    """
    #options_lst = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
    options_lst = ['1. ', '2. ', '3. ', '4. ', '5. ']
    alphabets_lst = ['A', 'B', 'C', 'D', 'E']

    def __init__(self, qn_text, qn_A, qn_B, qn_C, qn_D, qn_E, correct_ans):
        self.qn_text = qn_text
        self.a = qn_A
        self.b = qn_B
        self.c = qn_C
        self.d = qn_D
        self.e = qn_E
        self.correct_ans = correct_ans

        # The answer is NOT always A
        # So we need to randomise answer placement
        if correct_ans == 'A':
            self.ans_index = random.randint(0,4)
            self.qn_lst = [qn_B, qn_C, qn_D, qn_E]
            self.qn_lst.insert(self.ans_index, qn_A)
            self.qn_index = ['B', 'C', 'D', 'E']
            self.qn_index.insert(self.ans_index, 'A')
        else:
            self.qn_lst = [qn_A, qn_B, qn_C, qn_D, qn_E]
            self.ans_index = self.alphabets_lst.index(correct_ans)
            self.qn_index = ['A', 'B', 'C', 'D', 'E']
        self.ans_text = self.qn_lst[self.ans_index]
        self.qn_final_lst = []
        
        for i, opt in enumerate(self.qn_lst):
            single_opt = '%s%s' % (self.options_lst[i], opt)
            self.qn_final_lst.append(single_opt)
        
        self.ans = self.qn_final_lst[self.ans_index]

        self.options_string = ' '.join(self.qn_final_lst)
        self.qn = '%s %s Choose the correct number:' % (self.qn_text, self.options_string)
        self.gpt_response = None
        self.gpt_opt = None
        self.gpt_explanation = None
        self.gpt_cat = None
        self.response_index = None

    def validate(self, input_line):
        if input_line == self.ans_index+1 or input_line == str(self.ans_index+1):
            return True
        else:
            return False
        
    def test(self):
        if not self.gpt_response:
            self.gpt_response = get_chat_response(self.qn)
            find_first_digit = re.search("(\d)", self.gpt_response)
            if find_first_digit:
                first_digit = int(find_first_digit[1])
                self.gpt_opt = first_digit
                return self.gpt_opt
            else:
                return self.gpt_response
        else:
            print("Already got response!")

    def explain(self):
        if not self.gpt_response:
            self.test()
        
        if self.gpt_explanation:
            return self.gpt_explanation
        else:
            gpt_explain_prompt = 'Please explain why you chose this.'
            self.gpt_explanation = get_chat_response_3(self.qn, self.gpt_response, gpt_explain_prompt)
            return self.gpt_explanation
        
    def categorise(self):
        if self.gpt_cat:
            print("Already categorised!")
            return self.gpt_cat
        
        cat_prompt = 'Please categorise the following question: "%s"  What category would this question fall under? 1. Basic factual recall, either of basic science, physics, anatomy or regarding a procedure. 2. A question about clinical judgement, what is the correct diagnosis?' % self.qn_text
        self.gpt_cat = get_gpt_response(cat_prompt)
        return self.gpt_cat
    
    def orig_index(self):
        if not self.gpt_response:
            print("Please run qn.test() first!")
            return False
        
        response_no = int(self.gpt_opt)-1
        # The index should be something like [B, A, C, D, E]
        self.response_index = self.qn_index[response_no]
        return self.response_index



# Helper function to test a question and print out some raw output
# Need to give the class object predefined
# e.g.
#the_qn = test_qn(df_loc['Qn_Text'], df_loc['A'], df_loc['B'], df_loc['C'], 
#         df_loc['D'], df_loc['E'], df_loc['Ans'])

def test_a_qn(the_qn, explain=False):
    
    print("Testing question: %s" % the_qn.qn)
    gpt_ans = the_qn.test()
    print('AI chose option: "%s"' % gpt_ans, end = ' | ')
    print('Correct ans: "%s"' % the_qn.ans)
    marking = the_qn.validate(gpt_ans)

    if marking:
        print('Correct!')
    else:
        print('Wrong.')

    if explain:
        print('AI explanation: %s' % the_qn.explain()) 
    
    return marking

#%%
# Now we can run the script
# Start and end nos in case you don't want to run everything
start_no = 960    # INCLUSIVE
end_no = 978      # EXCLUSIVE
# Instantiating some lists to merge into a df later
test_qn_no = []
test_marking = []
test_explanation = []
test_response_index = []

# Import the csv
df_2a = pd.read_csv('Data/final_data2temp.csv')

for index, row in df_2a.loc[start_no: end_no].iterrows():
    the_qn = test_qn(row['Qn_Text'], row['A'], row['B'], row['C'], row['D'], row['E'], row['Ans'])
    print('[%d/%d]' % (index-start_no+1, end_no-start_no), end = ' ')
    mark = test_a_qn(the_qn, explain=True)
    gpt_ans_orig = the_qn.orig_index()
    test_qn_no.append(row['Qn'])
    test_marking.append(mark)
    test_explanation.append(the_qn.gpt_explanation)
    test_response_index.append(gpt_ans_orig)
    print('\n')

# Create and save the df
data_ans = {
    'GPT4_marking': test_marking,
    'GPT4_explain': test_explanation,
    'GPT4_ans_letter': test_response_index
}

df_2a_ans = pd.DataFrame(data_ans, index=test_qn_no)
print(df_2a_ans)

df_2a_ans.to_csv('temp_export_GPT4.csv')

# %%
