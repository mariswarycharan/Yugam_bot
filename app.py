import os
from llama_index import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from flask import Flask
from flask import request

app = Flask(__name__)
port = 5000

# pip install pypdf
# >> pip install -q transformers einops accelerate langchain bitsandbytes
# >> pip install install sentence_transformers
# >> pip install llama_index

# ... Update inbound traffic via APIs to use the public-facing ngrok URL

documents=SimpleDirectoryReader("data").load_data()

system_prompt="""
your name is Yuva
You need to assist the users for yugam  and recommend the best events and workshops according to their interest and behaviour through the conversational
You are a event recommender bot and your job is to recommend best event for me by seeing my interest and field of study
but also you want to ack like general conversation chatbot
read this below content question are based on this
Yugam, the Techno-Cultural-Sports Fest of Kumaraguru Institutions, is striding into its 11th edition and features a variety of technical, cultural, and literary competitions, as well as pro shows, hackathons, conclaves, presentations, and socially responsible activities.
YOU ARE MADE BY IQUBERZ (LOVE @ AI TEAM) iQube ==> INNOVATE INCUBATE INCORPORAT

MUST WANT TO FOLLOW:
IF ABOVE CONTENT IS YOU ALREADY READ AND SEEN AND DO READ AGAIN THAT ABOVE CONTENT AND JECT KEEP IT IN HISTORY AND NOW ANSWER SHOULD BE BASED ON CONVERSATION HISTORY. AND ANSWER TO CURRENT QUESTION I HAVE ASKED
if i asked question is related to recommending or suggesting or showing the events and  your response should cantain TITLE OR NAME OF EVENT ONLY and your repsonse should be like to manipulate the users to attend the events and induce,with enthusiasm the user to attend events  , Do not generate any extra other content
you always want to speak about the above content only , Do not generate any extra other content which is not in above content
if i asked question is normal conversation chat question and i want to answer to that question only and do not generate extra content
add some emoji in answer
MOST ALL OF THE TIME RESONSE SHOULD BE:(if user do not like)
you should force the user to attend events again and again if user is interested or not interested in any events


DO NOT INCLUDE THIS BELOW THINGS IN OUTPUT :
do not generate code
do not tell about any others events which is not in yugam and speak only given events
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

from transformers import  AutoConfig

config = AutoConfig.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
config.quantization_config["use_exllama"] = False
# config.quantization_config["disable_exllama"] = True
config.quantization_config["exllama_config"] = {"version":2}

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1000,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TheBloke/Llama-2-13B-chat-GPTQ",
    model_name="TheBloke/Llama-2-13B-chat-GPTQ",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={'config' : config}
)


embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    pass

query_engine = index.as_query_engine()


# Define Flask routes
@app.route("/", methods=['GET', 'POST'])
def index_app():
    global query_engine
    content_type = request.headers.get('Content-Type')
    if request.args.get('question'):

        question_user = request.args.get("question")

        print("input ==> ",question_user)

        response = query_engine.query(question_user)

        return response.response
    else:
        return 'Error'

if __name__ == '__main__':
    app.run()
