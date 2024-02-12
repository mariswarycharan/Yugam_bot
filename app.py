import os
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from flask import Flask
from flask import request
import json

app = Flask(__name__)
port = 5000

# pip install pypdf
# >> pip install -q transformers einops accelerate langchain bitsandbytes
# >> pip install install sentence_transformers
# >> pip install llama_index

# ... Update inbound traffic via APIs to use the public-facing ngrok URL

def qa_chain():

    prompt_template = """
    As a helpful assistant, I am excited to assist you with Yugam event recommendations and general conversation. I am programmed to suggest the best events and workshops tailored to your interests and field of study. I was developed by the iQuberz team (@ AI Team) at iQube, where we focus on INNOVATE, INCUBATE, and INCORPORATE.
    And i want to act like a general conversation chatbot and assistance related to Yugam
    
    Context : Yugam, the Techno-Cultural-Sports Fest of Kumaraguru Institutions, is gearing up for its 11th edition! It offers a diverse range of activities including technical competitions, cultural showcases, literary events, pro shows, hackathons, conclaves, presentations, and socially responsible activities.

    Please remember the following guidelines:
    - If you ask about event recommendations, I will provide the title or name of the event only. I will make the events appealing and induce enthusiasm in you to attend.
    - I will only discuss the events and content related to Yugam. I will not provide any information about other events or generate any extra content not in the provided context.
    - If you ask normal conversation chat questions, I will answer them directly without adding extra content.
    - I will use emojis to make conversations more engaging.
    - I will encourage you to attend events even if you are not initially interested.

    Context: {context}
    Question: {question}
    
    Answer: 
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["query","instruction"])
    
    embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")

    load_vector_store = Chroma(persist_directory="stores/yugam_cosine", embedding_function=embeddings)

    retriever = load_vector_store.as_retriever()

    llm = Together(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.7,
    max_tokens=300,
    top_k=1,

    together_api_key="40763e1166656125a452ff661e6218ac3d709fd64b458f17f94984acc8e748dc"
)

    chain_type_kwargs = {"prompt": prompt}
    
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
    )

    return qa

qa = qa_chain()


# Define Flask routes
@app.route("/", methods=['GET', 'POST'])
def index_app():
    global query_engine
    content_type = request.headers.get('Content-Type')
    if request.args.get('question'):

        question_user = request.args.get("question")

        print("input ==> ",question_user)

        try:
            response = qa(question_user)
            print(response['result'])
        except Exception as e:
            return f'Error in generating response: {e}'
        
        

        return response['result']
    else:
        return 'Error in api resquest' 

@app.route("/health" , methods=['GET'])
def health():
    if request.method == "GET":
        return json.dumps({'status': 'healthy'})
    else:
        return json.dumps({'status': 'unhealthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
