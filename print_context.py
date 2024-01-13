#%%
import json

# Print the relevant sources used for the answer
def print_docs(doc_list):
    for document in doc_list:
        if 'page' in document['metadata']:
            doc_page_no = f", page {document['metadata']['page']}/{document['metadata']['total_pages']}"
        else:
            doc_page_no = ""
        print(f"\n> {document['metadata']['source']}{doc_page_no}:")
        print(document['page_content'])

df_2a = pd.read_csv('pGPT_100.csv')

for index, row in df_2a.iterrows():
    if row['pGPT_ans_letter'] == 'X':
        print(row['Qn_Text'])
        print(f"A. {row['A']}")
        print(f"B. {row['B']}")
        print(f"C. {row['C']}")
        print(f"D. {row['D']}")
        print(f"E. {row['E']}")
        print(f"pGPT ans: {row['pGPT_ans_long']}")
        print("\n")
        print_docs(json.loads(row['pGPT_explain']))
# %%
