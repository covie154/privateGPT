#!/usr/bin/env python3
#%%
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.chat_models import ChatOpenAI
import os, time, re, random, json
import pandas as pd
import numpy as np

# First we get the variables from the .env file
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
model_n_gpu_layers = os.environ.get('MODEL_N_GPU_LAYERS')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
gpt_key = os.environ.get('GPT_key')

from constants import CHROMA_SETTINGS

# Now we set up the LLM
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()]
# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_ctx=model_n_ctx, n_batch=model_n_batch, n_gpu_layers=model_n_gpu_layers, callbacks=callbacks, verbose=True)
    case "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case "GPT4":
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=gpt_key, temperature=0)
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Print the relevant sources used for the answer
def print_docs(doc_list):
    for document in doc_list:
        if 'page' in document['metadata']:
            doc_page_no = f", page {document['metadata']['page']}/{document['metadata']['total_pages']}"
        else:
            doc_page_no = ""
        print(f"\n> {document['metadata']['source']}{doc_page_no}:")
        print(document['page_content'])

####################
# Question & Answer Time
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
        self.docs = {}
        self.docs_initial = {}
        self.response_index = None

    def validate(self, input_line):
        if input_line == self.ans_index+1 or input_line == str(self.ans_index+1):
            return True
        else:
            return False
        
    def test(self):
        if self.gpt_response:
            print("Already got response!")
            return self.gpt_response
        
        res = qa(self.qn)
        self.gpt_response, self.docs_initial = res['result'], res['source_documents']
        self.docs = json.dumps([{"page_content": i.page_content, "metadata": i.metadata} for i in self.docs_initial])

        find_first_digit = re.search("(\d)", self.gpt_response)
        if find_first_digit:
            first_digit = int(find_first_digit[1])
            self.gpt_opt = first_digit
            return self.gpt_opt
        else:
            self.gpt_opt = 99

    def explain(self):
        if not self.gpt_response:
            self.test()
        
        docs_json = json.loads(self.docs)
        print_docs(docs_json)
    
    def orig_index(self):
        if not self.gpt_response:
            print("Please run qn.test() first!")
            return False
        
        if self.gpt_opt > 5:
            return None
        
        response_no = int(self.gpt_opt)-1
        # The index should be something like [B, A, C, D, E]
        self.response_index = self.qn_index[response_no]
        return self.response_index
    
    
def test_a_qn(the_qn, explain=False):
    # Get the answer from the chain
    print("Testing question: %s" % the_qn.qn)
    start = time.time()
    gpt_ans = the_qn.test()
    print('AI chose option: "%s"' % gpt_ans, end = ' | ')
    print('Correct ans: "%s"' % the_qn.ans)
    marking = the_qn.validate(gpt_ans)
    end = time.time()

    if marking:
        print('Correct!')
    else:
        print('Wrong.')

    print(f'Elapsed time: {round(end - start, 2)} s')

    if explain:
        print('Sources:\n%s' % the_qn.explain()) 
    
    return marking

#%%
# Now we can run the script
# Start and end nos in case you don't want to run everything
start_no = 959    # INCLUSIVE
end_no = 979      # EXCLUSIVE
# Instantiating some lists to merge into a df later
test_qn_no = []
test_marking = []
test_explanation = []
test_response_index = []
test_orig_answer = []

# Import the csv
df_2a = pd.read_csv('final_data_w42.csv')

for index, row in df_2a.loc[start_no: end_no].iterrows():
    the_qn = test_qn(row['Qn_Text'], row['A'], row['B'], row['C'], row['D'], row['E'], row['Ans'])
    print('[%d/%d]' % (index-start_no+1, end_no-start_no), end = ' ')
    mark = test_a_qn(the_qn, explain=False)
    #if not mark:
    #    the_qn.explain()
    gpt_ans_orig = the_qn.orig_index()
    test_qn_no.append(row['Qn'])
    test_marking.append(mark)
    test_explanation.append(the_qn.docs)
    test_response_index.append(gpt_ans_orig)
    test_orig_answer.append(the_qn.gpt_response)
    print('\n')

# Create and save the df
data_ans = {
    'pGPT_marking': test_marking,
    'pGPT_explain': test_explanation,
    'pGPT_ans_letter': test_response_index,
    'pGPT_ans_long': test_orig_answer
}

df_2a_ans = pd.DataFrame(data_ans, index=test_qn_no)
print(df_2a_ans)

df_2a_ans.to_csv('temp_export_pGPT.csv')

#%%