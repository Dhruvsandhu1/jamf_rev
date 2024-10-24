import openai
import json
import streamlit as st
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
import os
import re

# Set your OpenAI API key
key1=st.secrets["OPENAI_API_KEY"]
API_KEY = os.getenv("OPENAI_API_KEY", key1)  # Replace if testing locally
OPENAI_API_KEY = key1
openai.api_key = key1


llm = ChatOpenAI(
    model="gpt-4",  # Use 'ChatOpenAI' for gpt-3.5-turbo
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

st.title("Jamf Reviews")
company_name = st.text_input("Enter your Company name", placeholder="perferably url",value="")
role=st.text_input("Enter your role", placeholder="eg: developer,analyst",value="")
domain=st.text_input("Which sector do you mainly operate in", placeholder="eg: Education,Legal,Security",value="")
specific=st.text_input("Any specific thing that you are looking from this tool", placeholder="eg: Customer support",value="")
company_size = st.selectbox(
    "Enter the size of the company",
    ["", "1-100", "101-1000", "1001-5000", "5000+"],
    format_func=lambda x: "Select" if x == "" else str(x)
)

def read_json_file(file_path):
    """Read the input JSON file and return its contents."""
    with open(file_path, 'r',encoding='utf-8') as f:
        return json.load(f)

def analyze_sentiment(text):
    system_prompt = f'''
    I am making a Customer Assist AI agent for enterprise SaaS sales.
    You have a customer profile and given a review ,based on the profile filter out wheather the review matches the customer profile or not.
    if the review is relevant then output 0 or else output 1.
    
    the customer profile is:
    {{p1}}
    {{p2}}
    {{p3}}
    {{p4}}
    {{p5}}


    **Relevance Criteria:**  

    
    A review is **relevant** if it aligns with atleast one of the following:  
    {{p6}}
    {{p7}}
    {{p8}}
    {{p9}}
    {{p10}}

    
    Example for a customer profile is given below
   {{p11}}
   {{p12}}
   {{p13}}
   {{p14}}
   {{p15}}
    

    Output 0 for relevant reviews and output 1 for irrelevant reviews
    output should be a single digit ie either 0 or 1 nothing else.

    '''

    customer_profile=[    
    f"**Company**: {company_name}",
    f"**Role**: {role}",
    f"**Sector**: {domain}",
    f"**Company Size**: {company_size}",
    f"**Specific Needs from Jamf**: {specific}"]


    relevance_criteria=[ "**Mentions the company** or shows awareness of the company’s needs",  
    "**Highlights benefits or challenges** related to the role (or a similar role)",  
    "**Addresses the sector**, showing how Jamf performs in that domain",  
    "**References the company size**, especially in relation to product fit or scalability",  
    "**Responds to specific needs** the customer is looking for in Jamf"]
    
    eg_criteria=["if the company in which the customer works is greenlight then output reviews that has some mentions of greenlight",
    "if their role is developer then output only those reviews which tells how is jamf for a developer or reviews that a developers must be shown about jamf",
    "if the sector they operate in is education then output only those reviews which tell about how jamf is performing in that domain or how is it bringing revolution in that domain",
    "if the company size is 5000+ then output only those reviews that have mentions of whether jamf is the fit for that organization or how jamf is playing the role in terms of scalability",
    "if specific thing they are looking in jamf is customer support then output reviews that has some information about customer support"]
    
    if company_name != "":
        system_prompt=system_prompt.replace(f'{{p1}}',f"{customer_profile[0]}")
        system_prompt=system_prompt.replace(f'{{p6}}',f"{relevance_criteria[0]}")
        system_prompt=system_prompt.replace(f'{{p11}}',f"{eg_criteria[0]}")

    if role != "":
        system_prompt=system_prompt.replace(f'{{p2}}',f"{customer_profile[1]}")
        system_prompt=system_prompt.replace(f'{{p7}}',f"{relevance_criteria[1]}")
        system_prompt=system_prompt.replace(f'{{p12}}',f"{eg_criteria[1]}")

    if domain != "":
        system_prompt=system_prompt.replace(f'{{p3}}',f"{customer_profile[2]}")
        system_prompt=system_prompt.replace(f'{{p8}}',f"{relevance_criteria[2]}")
        system_prompt=system_prompt.replace(f'{{p13}}',f"{eg_criteria[2]}")

    if company_size != "":
        system_prompt=system_prompt.replace(f'{{p4}}',f"{customer_profile[3]}")
        system_prompt=system_prompt.replace(f'{{p9}}',f"{relevance_criteria[3]}")
        system_prompt=system_prompt.replace(f'{{p14}}',f"{eg_criteria[3]}")

    if specific != "":
        system_prompt=system_prompt.replace(f'{{p5}}',f"{customer_profile[4]}")
        system_prompt=system_prompt.replace(f'{{p10}}',f"{relevance_criteria[4]}")
        system_prompt=system_prompt.replace(f'{{p15}}',f"{eg_criteria[4]}")
    
    cleaned_system_prompt = re.sub(r'\{p\d+\}', '', system_prompt)

    # st.write(cleaned_system_prompt)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" or "gpt-4" based on your need
        messages=[
            {"role": "system", "content": cleaned_system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.4
    )
    return response['choices'][0]['message']['content'].strip()


chunks_prompt1 = '''
Summarize the following text 
Text: {text}
'''

final_prompt = '''
Provide the final high quality refined summary of the entire document.
also the summary should be in respect to jamf.
It should only revolve around the customer profile
for example if the customer profile is "**Specific Needs from Jamf**: Customer support" then the summary should revolve around the customer support only and nothing else

Also provide the fitment rating out of 10.

the customer profile is :
    {{1p}}
    {{2p}}
    {{3p}}
    {{4p}}
    {{5p}}
Document: {text}

Output should be a summary and then a fitment rating and then the reason for that rating 
'''
customer_profile = [
    f"**Company**: {company_name}" if company_name else "",
    f"**Role**: {role}" if role else "",
    f"**Sector**: {domain}" if domain else "",
    f"**Company Size**: {company_size}" if company_size else "",
    f"**Specific Needs from Jamf**: {specific}" if specific else ""
]
system_prompt=final_prompt
for i, profile in enumerate(customer_profile,1):
    system_prompt = system_prompt.replace(f'{{{{{i}p}}}}', profile)  
for i in range(1, 6):
    system_prompt = system_prompt.replace(f"{{{{{i}p}}}}", "")
map_prompt_template = PromptTemplate(input_variables=["text"], template=chunks_prompt1)
final_prompt_template = PromptTemplate(input_variables=["text"], template=system_prompt)

answers = []
if st.button("Fetch the relevant reviews"):
    data1 = read_json_file("jamf_senti.json" )
    data2 = read_json_file("ampify_data.json")
    data=data1+data2
    output_file='combined.json'
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
    data=read_json_file('combined.json')
    i = 1
    for entry in data:
        if entry['useful']=='0':
            # Handle posts
            if 'title' in entry:  # We are dealing with a post
                text = entry["body"]
                if entry['body']=="":
                    text=entry['title']
                customer_useful = analyze_sentiment(text)
                temp_dict={
                    "title": entry['title'],
                    "author": entry['author'],
                    "url": entry['url'],
                    "body": entry['body'],
                    "useful": entry['useful'],
                    "sentiment": entry['sentiment'],
                    "customer_useful": customer_useful
                }
                if 'subreddit' in entry:
                    temp_dict['subreddit']=entry['subreddit']
                if 'created' in entry:
                    temp_dict['created']=entry['created']
                if 'upvotes' in entry:
                    temp_dict['upvotes']=entry['upvotes']
                if 'rating' in entry:
                    temp_dict['rating']=entry['rating']
                if 'platform' in entry:
                    temp_dict['platform']=entry['platform']
                answers.append(temp_dict)
            # Handle comments within posts
            else:
                text = entry['body']
                customer_useful= analyze_sentiment(text)
                answers.append({
                    'author': entry['author'],
                    'url': entry['url'],
                    "body": entry['body'],
                    "subreddit": entry['subreddit'],
                    "useful": entry['useful'],
                    "sentiment": entry['sentiment'],
                    "customer_useful": customer_useful,
                    "created": entry['created'],
                    "upvotes":entry['upvotes']
                })
        i += 1
    
    # Save sentiment insights to a JSON file
    with open("customer.json", "w") as json_file:
        json.dump(answers, json_file, indent=4)
    st.write("reviews have been saved to 'customer.json'.")
    with open('customer.json', 'r') as f:
        data=json.load(f)
    chunks=[]
    for item in data:
        if item['useful']=='0' and item['customer_useful']=='0':
            chunks.append(item['body'])
    refined_chunks=[]
    for i in range(len(chunks)):
        refined_chunks.append(Document(page_content=str(chunks[i])))
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True
    )
    # Optional: Display some relevant reviews in Streamlit
    st.title("Summary")
    summary = summary_chain.run(refined_chunks)
    st.write(summary)
    st.markdown("______________________")
    st.title("Some reviews that matches your profile are")
    with open("customer.json", "r") as json_file:
        file = json.load(json_file)
    for elem in file:
        flag=0
        if elem['useful'] == '0' and elem['customer_useful']=='0':  # Display reviews marked as '0'
            if elem['body']=="":
                flag=1
            if 'title' in elem and flag==0:
                st.markdown(elem['title'])
            if 'platform' in elem:
                 st.markdown(f"Platform : G2 | [Open Review]({elem['url']})")
            else:   
                st.markdown(f"Platform : Reddit/{elem['subreddit']} | {elem['created'].split()[0]} | [Open Review]({elem['url']}) | Upvotes : {elem['upvotes']}")
            sample_para=elem['body']
            if flag==1:
                sample_para=elem['title']
            para = sample_para.replace('\n', '<br>')
            if elem['sentiment']=='2':
                st.markdown(
                            f"""
                            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                                <p style="color: #155724; font-size: 16px; margin: 0;">{para}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            elif elem['sentiment']=='4':
                st.markdown(
                            f"""
                            <div style="background-color: #cce5ff; padding: 10px; border-radius: 5px;">
                                <p style="color: #004085; font-size: 16px;">{para}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(
                            f"""
                            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                                <p style="color: #721c24; font-size: 16px;">{para}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown("______________________")
    