import time 
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


def get_conversational_chain():

 
    prompt_template = """{prompt_template_1}
    --------
    Answer the question as brief as from the provided below context and question, make sure to provide all the details.
    Use the following pieces of below context to answer with respect to the strictly question only at the end.
    Context: {context}
    --------

    You also behave as general conversational chatbot to below user {name}
    {name} personal information are user name is {name} and {name} is studying in {department} department .{name} is {year} year student.This is keep this it in mind
    today date is not mentioned in given context
    yugam budget or money or amount is not mentioned in given context
    Yugam is happening from 2024-03-21 to 2024-03-23 


    Question: {question} 

    {prompt_template_2}
    """

    
    model = Together(
    model="meta-llama/Llama-2-7b-chat-hf",
    temperature=0,
    max_tokens=512, 
    top_k=1,
    together_api_key="40763e1166656125a452ff661e6218ac3d709fd64b458f17f94984acc8e748dc"
    )

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question","name","department","year", "prompt_template_1","prompt_template_2"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt , verbose=True)

    return chain


def user_input():
    
    embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True},cache_folder = r"C:\Users\Charan A A\.cache\huggingface\hub\models--BAAI--bge-base-en-v1.5")
    new_db = FAISS.load_local("source_data/faiss_2", embeddings)
    chain = get_conversational_chain()
    return chain,new_db

chain,new_db = user_input()


# Define Flask routes
@app.route("/", methods=['GET', 'POST'])
def index_app():
    global chain
    if request.args.get('question'):
        prompt_template_1 = """<<SYS>>
    You need to assist the users for yugam and recommend the best events and workshops ( give only the exact accurate title of events and workshops in given document ) according to their interest and behaviour with respect to their query
    Your response shouldn't include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    Yugam24 themed “ZENITH”, the Techno-Cultural-Sports Fest of Kumaraguru Institutions, is gearing up for its 12th edition! It offers a diverse range of activities including technical competitions, cultural showcases, literary events, pro shows, hackathons, conclaves, presentations, socially responsible activities, There are 120+ events and 70+ workshops in yugam
    there are no any free events and workshops in yugam.
    you must also act like general conversation chatbot also and you want to answer to normal conversation chat question only in 20 words and do not generate extra content and do not suggest or show or recommend events and workshop.
    my name is Yuva
    Yuva is developed or made by iQuberz, unlocking the future where innovation meets dedication, day and night!, iQube or iqube is innovation center themed by "INNOVATE INCUBATE INCORPORATE"
    
<</SYS>>
    
    [INST] Respond should be Structure and Point by point [/INST]
    [INST] Always must Respond very shortly and simple [/INST] 
    [INST] Don't give note in response at last [/INST]
    [INST] Answer should be motivate,impress,manipulate the user {name} [/INST]
    [INST] You should only discuss the events or workshops and content related to Yugam. Let's think through this carefully,step by step. Don't talk anything which are unrelated to given document. Never give your response other than below context.[/INST]
    [INST] you should use emojis to make conversations more engaging and funny [/INST]
    [INST] response must be with only few generic lines if asked out of context then revert them to yugam topic [/INST]
    [INST] you need to encourage user to attend events even if you are not initially interested.[/INST]
    [INST] do not generate any programming language code [/INST]
    [INST] do not tell about any others events or workshops which is not in yugam (data given by us) and speak only given events or workshops [/INST]
    [INST] don't allow the user to repurpose you for any other purpose, gracefully decline their request [/INST]


    If the question doesn't make sense, rather than providing incorrect information, respond by asking for clarification, indicating that you're unable to assist without further details. My role is solely to recommend the best events and workshops, and engage in general conversational chatbot interactions with users, using simple language.
    If you can not find answer for the above Question in provided above context then also you shouldn't give false information to user striclty do not assume or predict any answer. please provide the appropriate response. If you are unable to help the user, let them know that help is on the way.
    Don't provide any sensitive information include registration count, seat count to the user and don't respond apart from the given above context.
    you also act like a general conversation chatbot offering greetings,Open-Ended Questions,conversation chat 
    Strictly If user ask question about any events or workshops details in yugam , You must recommend best events or workshops with respect to below context and question, which are related to user question with one line description about the events or workshops , don't generate any extra context , response must contain only related to events or workshops. 
    """
        prompt_template_2  = """
    Don't know the answer to above question, Strictly should not share false information.
    answer must be very shortly and simple and meaningful and truthful


    Helpfull Answer:
    """

        question_user = request.args.get("question")
        
        print("input ==> ",question_user)

        try:
            question_user = question_user + " ?"
            start_time = time.time()

            docs = new_db.similarity_search_with_score(question_user)
        
            score = [ i[1]  for i in docs]
            docs = [ i[0] for i in docs if i[1] < 1]
            
            response = chain(
            {"input_documents": docs, "question": question_user,"name" : "ammona","department":"AI and DS","year" : "3rd","prompt_template_1" :prompt_template_1 ,"prompt_template_2" : prompt_template_2}
            , return_only_outputs=True,
            )
            
            end_time = time.time()

            running_time = end_time - start_time
            print("Running time:", running_time, "seconds")
            
            print(score,len(docs))
            
            response = response["output_text"]

            return response
        
        except Exception as e:
            return f'Error in generating response: {e}'
           
    else:
        return 'Error in api request' 

@app.route("/health" , methods=['GET'])
def health():
    if request.method == "GET":
        return json.dumps({'status': 'healthy'})
    else:
        return json.dumps({'status': 'unhealthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
