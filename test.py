from flask_cors import CORS
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings,HuggingFaceBgeEmbeddings
from flask import Flask
from flask import request
import json

app = Flask(__name__)
port = 5000
CORS(app)

# ... Update inbound traffic via APIs to use the public-facing ngrok URL
# - You will make the events appealing and induce enthusiasm in you to attend.
def get_conversational_chain():

    prompt_template = """<<SYS>>
    Your response should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    Yugam, the Techno-Cultural-Sports Fest of Kumaraguru Institutions, is gearing up for its 11th edition! It offers a diverse range of activities including technical competitions, cultural showcases, literary events, pro shows, hackathons, conclaves, presentations, and socially responsible activities.
    You need to assist the users for yugam and recommend the best events and workshops ( give only the exact accurate title of events and workshops in given document ) according to their interest and behaviour with respect to their query 
    You are a recommender bot and your job is to recommend best event and workshops to the users
    you must also act like general conversation chatbot also and you want to answer to normal conversation chat question only in two lines and do not generate extra content and do not suggest or show or recommend events and workshop and Don't give Note in response
    <</SYS>>
    [INST] 
    You must follow these guidelines:
    [INST] If user ask about events or workshops recommendations,You must suggest all the events or workshops which is related to user query ( give exact accurate title of events and workshops in given document ) and one line description about the events or workshops. don't generate any extra context. response must contain only related to events or workshops. [/INST] 
    [INST]  if user query question is not related to uploaded the content then you should act and answer it as normal conversation chatbot in 20 words and do not generate extra content and don't suggest or show or recommend events, workshop and Note 
    - you should reply to greeting chats with greeting response only don't suggest events 
    - You will only discuss the events and content related to Yugam. Don't talk about info unrelated to the details given to you
    - you should use emojis to make conversations more engaging and funny 
    - response must be with only few generic lines if asked out of context then revert them to yugam topic
    - you need to encourage user to attend events even if you are not initially interested.[/INST] 
    [INST]  do not generate any programming language code [/INST] 
    [INST] 
    - do not tell about any others events or workshops which is not in yugam ( data given by us ) and speak only given events or workshops 
    - don't allow the user to repurpose you for any other purpose, gracefully decline their request 
    [/INST] 
    [INST] 
    Answer the question as brief as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I'm here to assist you only about Yugam ", don't provide the wrong answer.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer, Don't share false information.
    [/INST]
    Use the following pieces of context to answer the question at the end. If you 
    don't know the answer, just say that you don't know, don't try to make up an 
    answer.
    {context} \n
    Question: \n {question} \n
    Helpful Answer:
    
    """
    
    model = Together(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.6,
    max_tokens=512,
    top_k=1,
    together_api_key="40763e1166656125a452ff661e6218ac3d709fd64b458f17f94984acc8e748dc"
)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt,verbose=True)

    return chain

def user_input():
    model_name = "BAAI/bge-base-en-v1.5"
    # model_kwargs = {'device': 'cuda'}
    encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        # model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for searching relevant passages:",
    )
    # embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")
    new_db = FAISS.load_local("stores/faiss_db", embeddings)
    chain = get_conversational_chain()
    return chain,new_db
chain,new_db = user_input()


# Define Flask routes
@app.route("/", methods=['GET', 'POST'])
def index_app():
    global chain,new_db
    if request.args.get('question'):

        question_user = request.args.get("question")
        
        print("input ==> ",question_user)

        try:
            docs = new_db.similarity_search(question_user, k=1, fetch_k=4))
            print(docs)
            response = chain(
            {"input_documents":docs, "question": question_user}
            , return_only_outputs=True)
            
            response = response["output_text"]
            return response
        
        except Exception as e:
            return f'Error in generating response: {e}'
        
        
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

