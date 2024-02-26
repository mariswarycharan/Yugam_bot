from celery import shared_task
from celery.task import task
from django.conf import settings
from sentry_sdk import capture_exception
import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS


@task(name="updateFaissDB")
def updateFaissDB():
    try:
        from events.models import Event,Category,SubCategory
        from workshop.models import Workshop
        from userdashboard.models import YugamConfig
        try:
            gemini_key = YugamConfig.objects.get(key='gemini_key').value
        except:
            gemini_key = ""

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-pro')

        events = pd.DataFrame(list(Event.objects.all().values()))
        workshops = pd.DataFrame(list(Workshop.objects.all().values()))   
        categories = pd.DataFrame(list(Category.objects.all().values()))
        subCategories = pd.DataFrame(list(SubCategory.objects.all().values()))

        titles = []
        descriptions = []
        generated_descriptions = []
        sub_cato_list = []
        cato_list = []
        des_sub_cato_list = []
        keywords_list = []
        image_url_list = []
        events_url_list = []
        events_tag_list = []
        titles = []

        for i in tqdm(range(len(events))):
            html_code = events["description"][i]
            soup = BeautifulSoup(html_code, 'html.parser')
            text_content = soup.get_text(separator="\n", strip=True)
            
            try:
                prompt_gemini = f"""
                ANTICIPATE ATTENDEE FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY FOR BELOW EVENT PLANNING. UNDERSTANDING WHO IS LIKELY TO ATTEND AND WHAT DEPARTMENTS ARE INTERESTED IN WORKSHOPS ENHANCES ENGAGEMENT AND TAILORS CONTENT TO MEET DIVERSE NEEDS EFFECTIVELY.        
                ADD THIS ALL FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY INDIVIDUALLY IN ANSWER
                
                AND ADD TO THE ANSWER WHICH AREA OF INTEREST IS PERFECTLY SUITABLE FOR THIS BELOW EVENTS ADD TOP 3. AREA OF INTEREST  ARE (mechanical, computing ,electrical ,electronics ,life science ,literature ,management, Finance ,Liberal Arts, fine Arts, media ,civil, leadership ,entrepreneurship, Sports, performing arts )  
                ADD AREA OF INTEREST NAME ONLY
                
                AND ADD TO THE ANSWER WHICH DOMAIN IS PERFECTLY SUITABLE FOR THIS BELOW EVENTS ADD TOP 3. DOMAINS  ARE ( Web Development, Mobile App Development, Software Engineering, Data Science, Artificial Intelligence, Machine Learning, Cybersecurity, Cloud Computing, Game Development, Database Management, Networking  DevOps, UI/UX Design, Graphic Design, Digital Marketing, Content Writing, Business Analysis
                Embedded Systems,Robotics,Internet of Things (IoT),Sensor,Signal,mechanical,Mechatronics)
                ADD DOMAIN NAME ONLY
                
                EVENTS DETAILS ARE 
                TITLE IS {events['title'][i]}
                DECRIPTION IS {text_content}

                """
                palm = model.generate_content(prompt_gemini)
                generated_des = palm.text
            except:
                generated_des = re.sub(r'["\[,\]\\]', ' ', events.loc[i]['event_tags'])
                
            value = subCategories.loc[subCategories['id'] == events["subCategory_id"][i]]
            sub_cato = value.iloc[0]['name']
            des_sub_cato =  value.iloc[0]['description']
            value1 = categories.loc[categories['id'] == value.iloc[0]['category_id']]
            cata = value1.iloc[0]['name']
            titles.append(events['title'][i])
            descriptions.append(text_content)
            generated_descriptions.append(generated_des)
            sub_cato_list.append(sub_cato)
            des_sub_cato_list.append(des_sub_cato)
            cato_list.append(cata)
            image_url_list.append("https://yugam.in/media/" +  events.loc[i]['image'])
            events_url_list.append("https://yugam.in/e/" +  events.loc[i]['event_url'])
            events_tag_list.append(re.sub(r'["\[,\]\\]', ' ', events.loc[i]['event_tags']))
            
        data = {'Title': titles, 'Description': descriptions , "generated_descriptions" : generated_descriptions , "catagory" : cato_list, "sub_catagory" :sub_cato_list , 
            "description_of_sub_catagory" : des_sub_cato_list ,'image_url' : image_url_list , 'events_url' : events_url_list , 'events_tags' : events_tag_list  }

        events_filtered_df = pd.DataFrame(data)
 

        for i in tqdm(range(len(workshops))):
            html_code = workshops["description"][i]
            soup = BeautifulSoup(html_code, 'html.parser')
            text_content = soup.get_text(separator="\n", strip=True)
            try:
                prompt_gemini = f"""
                ANTICIPATE ATTENDEE FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY FOR EVENT PLANNING. UNDERSTANDING WHO IS LIKELY TO ATTEND AND WHAT DEPARTMENTS ARE INTERESTED IN WORKSHOPS ENHANCES ENGAGEMENT AND TAILORS CONTENT TO MEET DIVERSE NEEDS EFFECTIVELY.        
                ADD THIS ALL FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY INDIVIDUALLY IN ANSWER
                
                ADD ADD TO THE ANSWER WHICH AREA OF INTEREST IS PERFECTLY SUITABLE FOR THIS EVENTS ADD TOP 3 . AREA OF INTEREST  ARE (mechanical, computing ,electrical ,electronics ,life science ,literature ,management, Finance ,Liberal Arts, fine Arts, media ,civil, leadership ,entrepreneurship, Sports, performing arts )  
                ADD AREA OF INTEREST NAME ONLY
                
                AND ADD TO THE ANSWER WHICH DOMAIN IS PERFECTLY SUITABLE FOR THIS BELOW WORKSHOP ADD TOP 3. DOMAINS  ARE ( Web Development, Mobile App Development, Software Engineering, Data Science, Artificial Intelligence, Machine Learning, Cybersecurity, Cloud Computing, Game Development, Database Management, Networking  DevOps, UI/UX Design, Graphic Design, Digital Marketing, Content Writing, Business Analysis
                Embedded Systems,Robotics,Internet of Things (IoT),Sensor,Signal,mechanical,Mechatronics)
                ADD DOMAIN NAME ONLY

                WORKSHOP DETAILS ARE 
                TITLE IS {events['title'][i]}
                DECRIPTION IS {text_content}

                """
                palm = model.generate_content(prompt_gemini)
                generated_des = palm.text
            except:
                generated_des = re.sub(r'["\[,\]\\]', ' ', workshops.loc[i]['workshop_tags'])

            
            value = subCategories.loc[subCategories['id'] == workshops["subCategory_id"][i]]
            sub_cato = value.iloc[0]['name']
            des_sub_cato =  value.iloc[0]['description']
            value1 = categories.loc[categories['id'] == value.iloc[0]['category_id']]
            cata = value1.iloc[0]['name']
            

            titles.append(workshops['title'][i])
            descriptions.append(text_content)
            generated_descriptions.append(generated_des)
            sub_cato_list.append(sub_cato)
            des_sub_cato_list.append(des_sub_cato)
            cato_list.append(cata)
            image_url_list.append("https://yugam.in/media/" +  workshops.loc[i]['image'])
            events_url_list.append("https://yugam.in/w/" +  workshops.loc[i]['workshop_url'])
            events_tag_list.append(re.sub(r'["\[,\]\\]', ' ', workshops.loc[i]['workshop_tags']))
        
        data = {'Title': titles, 'Description': descriptions , "generated_descriptions" : generated_descriptions 
                , "catagory" : cato_list, "sub_catagory" :sub_cato_list ,"description_of_sub_catagory" : des_sub_cato_list , 
                'image_url' : image_url_list , 'events_url' : events_url_list , 'events_tags' : events_tag_list }
        workshops_filtered_df = pd.DataFrame(data)


        text = ''
        for i in range(len(events_filtered_df)):
            val = events_filtered_df.loc[i]
            title_name = '" ' + val['Title'].upper() + ' "'
            text +=  "TITLE of the event is " + title_name + " and "
            text +=  "description of the "+ title_name +" is " + val['Description'].replace("\r\n",'').replace('\n','')  + "\n"
            text +=  "Anticipate attendee for " + title_name + "event have " + val['generated_descriptions'].replace('\n','').replace('*','')  + " and "
            text +=  "category is " + val['catagory'] + " and "
            text +=  "sub category is " + val['sub_catagory'] + ' and '
            text +=  "url : " + val['events_url'] + ' and '
            text +=  "events tags are " + val['events_tags'] + '\n'

            # text +=  "description of the sub catagory of the "+ title_name +" is " + val['description_of_sub_catagory'] + " "
            # text +=  "key terms of the "+ title_name +" are " + str(val['keywords'] )+ '\n'
            
        with open('yugamAI/ai_database/events_content_yugam.txt', 'w', encoding='utf-8') as file:
            file.write(text)
            
        text = ''
        for i in range(len(workshops_filtered_df)):
            val = workshops_filtered_df.loc[i]
            workshop_title_name = '" ' + val['Title'].upper()+ ' "'

            text +=  "TITLE of the workshop is " + workshop_title_name + " and "
            text +=  "description of the " + workshop_title_name + " is " + val['Description'].replace("\r\n",'').replace('\n','') + "\n"
            text +=   "Anticipate attendee for " + workshop_title_name + "workshop have " + val['generated_descriptions'].replace('\n','').replace('\r\n','').replace('*','')  +" and "
            text +=  "category is " + val['catagory'] + " and "
            text +=  "sub category is" + val['sub_catagory'] + ' and '
            text +=  "url : " + val['events_url'] + ' and '
            text +=  "workshops tags are " + val['events_tags'] + '\n'
            
        with open('yugamAI/ai_database/workshops_content_yugam.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        
        text = ''
        loader = DirectoryLoader('yugamAI/ai_database/', glob="**/*.txt", show_progress=True, loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()
        print(documents)
        text += documents[0].page_content + "\n\n"
        text += documents[1].page_content 

        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=1, chunk_overlap=1)
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True},)
        
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("yugamAI/ai_database/faiss")
        
    except Exception as e:
        print(e)
