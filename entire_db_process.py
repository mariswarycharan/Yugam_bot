import psycopg2
import pandas as pd
import requests,re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


def updateFaissDB():
     
    all_table_names_list = ['django_migrations', 'django_content_type', 'auth_permission', 'auth_group', 'auth_group_permissions', 'auth_user', 'auth_user_groups', 'auth_user_user_permissions', 'django_admin_log', 'notifications_newsfeed', 'notifications_slides', 'online_yugam_webinar', 'online_yugam_meeting', 'events_category', 'events_event', 'online_yugam_teamsrefreshtoken', 'userdashboard_department', 'events_event_admins', 'payment_deletedtransaction', 'userdashboard_eventteam', 'events_eventround_teams', 'events_subcategory', 'events_winners_teams', 'workshop_workshop_admins', 'workshop_workshopattendance', 'workshop_workshop', 'admins_eventpointsmapping', 'admins_blacklist', 'payment_onlinerefund', 'admins_rooms', 'admins_booking', 'admins_venue', 'admins_venuebooking', 'products', 'products_stockentry', 'userdashboard_college', 'sendmail_approveeventcertificate', 'userdashboard_fileupload', 'userdashboard_answers', 'discord_discordchannel', 'discord_discordchannelparticipant', 'sendmail_approveworkshopcertificate', 'sendmail_eventemaillog', 'sendmail_webhookslog', 'webcontent_contact', 'sendmail_whatsappmessagelog', 'events_combo', 'payment_cardpayment', 'payment_cash', 'sendmail_workshopemaillog', 'sendmail_approvalemail', 'webcontent_studentambassadorregistraion', 'webcontent_yugam360_hash', 'django_session', 'sms_eventsmslog', 'sms_eventmarketingwhatsapplog', 'sms_approvalmarketing', 'sms_workshopmarketingwhatsapplog', 'sms_workshopsmslog', 'userdashboard_eventcertificatelog', 'social_auth_association', 'social_auth_code', 'social_auth_nonce', 'userdashboard_eventregistration', 'social_auth_usersocialauth', 'social_auth_partial', 'userdashboard_paperpresentation', 'uniquecodes_maprfidbarcode', 'events_eventround_participants', 'userdashboard_bankdetails', 'events_winners_participants', 'userdashboard_school', 'webcontent_sponsor', 'userdashboard_state', 'userdashboard_profile', 'webcontent_templateimage', 'userdashboard_trendingworkshop', 'webcontent_yugam360', 'userdashboard_tokens', 'userdashboard_trendingevent', 'userdashboard_workshopcertificatelog', 'webcontent_seat', 'userdashboard_yugamconfig', 'payment_transaction', 'userdashboard_workshopregistration', 'webcontent_yugam360booking', 'userdashboard_workshopteam', 'webcontent_yugam360image', 'webcontent_yugam360vipbooking', 'dynamicforms_formresponse', 'dynamicforms_payment_form_actions', 'dynamicforms_question', 'dynamicforms_answer', 'events_combo_seats', 'events_combo_wrshps', 'dynamicforms_form', 'events_combodynamic_excluded_workshop', 'events_combodynamic', 'yugamtheme_whatsapptoken', 'events_comboregistration', 'events_combo_events', 'events_eventcombo', 'events_seatcombo', 'events_eventround', 'events_workshopcombo', 'events_winners', 'events_domain']

    def sql_to_dataframe(table_name):
        conn = psycopg2.connect(
                    database="postgres",
                    user="postgres",
                    password="nuttertools@123",
                    host="10.1.76.58",
                    port="5436"
                    )
        
        cursor = conn.cursor()
        columns_query = "SELECT column_name FROM information_schema.columns WHERE table_name = '" + table_name + "'"
        cursor.execute(columns_query)
        table_columns_name = cursor.fetchall()
        table_columns_name = [ i[0] for i in table_columns_name]
                
        try:
            query = "SELECT * FROM " + table_name
            cursor.execute(query=query)

        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)

        # The execute returns a list of tuples:
        tuples_list = cursor.fetchall()
        

        cursor.close()
        # Now we need to transform the list into a pandas DataFrame:
        df = pd.DataFrame(tuples_list,columns=table_columns_name)
        return df


    def gemini_request(query,api_index):
        API_KEY_LIST = ['AIzaSyBHNjNmTdTZOwLr0ucsiZowtLZkI2U3ztM','AIzaSyBLbePjmaxgBIOo7I0Bh6o6Bq3FWvpO83I','AIzaSyBf1f92VARfojVOJee8KYziFDEwPg--2N4','AIzaSyDp_bPEbYnbFIXaQ99e4QKFkTawWptL7q0','AIzaSyBI5H2n6aKP7iYYn2PzSj5LSQ7DJHBxKJQ'
                        ,'AIzaSyD1mrEZHdZqt2bswUBKj2gBNe8wr5XVLXo','AIzaSyA21ny7hTPZCAZrQwYAoCrVqTJVbHVdBmw','AIzaSyB3ZKlIluPSdGKphLBzzC4XYuAZ55Uf9qs','AIzaSyC2zLA-jePiNqxNKx_gsBtkVtajm8PqWCM','AIzaSyDQCan0QQ7dWZpJzXPvmnQcc0vK3jAuzNQ']
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + API_KEY_LIST[api_index]
        data = {
            "contents": [{
                "parts": [{
                    "text": query
                }]
            }]
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                content_text = response_data.get('candidates', [])[0].get('content', {}).get('parts', [])[0].get('text', None)
                if content_text:
                    return content_text
                else:
                    return ''
            else:
                print("Request failed with status code:", response.status_code , "api_index ==> ",api_index)
                print("Response:", response.text)
                return ''
        except Exception as e:
            return ''
                
          
    events_df = sql_to_dataframe('events_event')
    categories_df = sql_to_dataframe('events_category')
    subCategories_df = sql_to_dataframe('events_subcategory')
    workshops_df = sql_to_dataframe('workshop_workshop')       
        
    full_text_events = ''
    
    for i in tqdm(range(len(events_df))):
        
        event_loc = events_df.loc[i]
        html_code = event_loc["description"]
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_code, 'html.parser')
        # Extract text content
        text_content_description = soup.get_text(separator="\n", strip=True)
        
        # events description generation function

        prompt_gemini = f"""
        ANTICIPATE ATTENDEE FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY FOR BELOW EVENT PLANNING. UNDERSTANDING WHO IS LIKELY TO ATTEND AND WHAT DEPARTMENTS ARE INTERESTED IN WORKSHOPS ENHANCES ENGAGEMENT AND TAILORS CONTENT TO MEET DIVERSE NEEDS EFFECTIVELY.        
        ADD THIS ALL FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY INDIVIDUALLY IN ANSWER
        
        AND ADD TO THE ANSWER WHICH AREA OF INTEREST IS PERFECTLY SUITABLE FOR THIS BELOW EVENTS ADD TOP 3. AREA OF INTEREST  ARE (mechanical, computing ,electrical ,electronics ,life science ,literature ,management, Finance ,Liberal Arts, fine Arts, media ,civil, leadership ,entrepreneurship, Sports, performing arts )  
        ADD AREA OF INTEREST NAME ONLY
        
        AND ADD TO THE ANSWER WHICH DOMAIN IS PERFECTLY SUITABLE FOR THIS BELOW EVENTS ADD TOP 3. DOMAINS  ARE ( Web Development, Mobile App Development, Software Engineering, Data Science, Artificial Intelligence, Machine Learning, Cybersecurity, Cloud Computing, Game Development, Database Management, Networking  DevOps, UI/UX Design, Graphic Design, Digital Marketing, Content Writing, Business Analysis
        Embedded Systems,Robotics,Internet of Things (IoT),Sensor,Signal,mechanical,Mechatronics)
        ADD DOMAIN NAME ONLY
        
        EVENT DETAILS:
        TITLE IS {event_loc['title']}
        DECRIPTION IS {text_content_description}

        """
        generated_description_gemini = gemini_request(prompt_gemini,api_index=int(str(i)[-1]))

        if generated_description_gemini == '' :
            print(generated_description_gemini)
            generated_description_gemini = re.sub(r'["\[,\]\\]', ' ', event_loc['event_tags'])
            
        value = subCategories_df.loc[subCategories_df['id'] == event_loc["subCategory_id"]]
        subCategory = value.iloc[0]['name']
        value1 = categories_df.loc[categories_df['id'] == value.iloc[0]['category_id']]
        Category = value1.iloc[0]['name']
        
        
        title_name = '" ' + '#YUG-E-' + str(event_loc['id'])  + event_loc['title'].upper()+ ' "'
        full_text_events +=  "TITLE of the event is " + title_name + " and "
        full_text_events +=  "description of the "+ title_name +" event is " + text_content_description.replace("\r\n",'').replace('\n',' ')  + "\n"
        full_text_events +=  "area of interests and skills and technology and domains are required for  " + title_name + " event are  " + generated_description_gemini.replace('\n',' ').replace('*','')  + " and "
        full_text_events +=  "CATEGORY is " + Category + " and "
        full_text_events +=  "SUB CATEGORY is " + subCategory + ' and '
        full_text_events +=  "register link or URL for " + title_name + " event is " + "https://yugam.in/e/" +  event_loc['event_url'] + ' and '
        full_text_events +=  title_name + " event happening on " +  str(event_loc['startTime']) + ' and '
        full_text_events +=  "WINNING PRICE AMOUNT FOR " + title_name + " event is " +  str(event_loc['price_amount']) + ' and '
        full_text_events +=  "ENTRY FEES OR COST OR REGISTRATION FEES FOR " + title_name + " event is " +  str(event_loc['common_paymentAmount']) + ' and '
        full_text_events +=  "events tags are " + re.sub(r'["\[,\]\\]', ' ', event_loc['event_tags']) + '\n'
        
    
    full_text_workshops = ''
    
    for i in tqdm(range(len(workshops_df))):
        
        workshop_loc = workshops_df.loc[i]
        html_code = workshop_loc["description"]
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_code, 'html.parser')
        # Extract text content
        text_content_description = soup.get_text(separator="\n", strip=True)
        
        # events description generation function
        prompt_gemini = f"""
        ANTICIPATE ATTENDEE FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY FOR EVENT PLANNING. UNDERSTANDING WHO IS LIKELY TO ATTEND AND WHAT DEPARTMENTS ARE INTERESTED IN WORKSHOPS ENHANCES ENGAGEMENT AND TAILORS CONTENT TO MEET DIVERSE NEEDS EFFECTIVELY.        
        ADD THIS ALL FIELD OF STUDY AND AREA OF INTERESTS AND SKILLS AND TECHNOLOGY INDIVIDUALLY IN ANSWER
        
        ADD ADD TO THE ANSWER WHICH AREA OF INTEREST IS PERFECTLY SUITABLE FOR THIS EVENTS ADD TOP 3 . AREA OF INTEREST  ARE (mechanical, computing ,electrical ,electronics ,life science ,literature ,management, Finance ,Liberal Arts, fine Arts, media ,civil, leadership ,entrepreneurship, Sports, performing arts )  
        ADD AREA OF INTEREST NAME ONLY
        
        AND ADD TO THE ANSWER WHICH DOMAIN IS PERFECTLY SUITABLE FOR THIS BELOW WORKSHOP ADD TOP 3. DOMAINS  ARE ( Web Development, Mobile App Development, Software Engineering, Data Science, Artificial Intelligence, Machine Learning, Cybersecurity, Cloud Computing, Game Development, Database Management, Networking  DevOps, UI/UX Design, Graphic Design, Digital Marketing, Content Writing, Business Analysis
        Embedded Systems,Robotics,Internet of Things (IoT),Sensor,Signal,mechanical,Mechatronics)
        ADD DOMAIN NAME ONLY

        WORKSHOP DETAILS:
        TITLE IS {workshop_loc['title']}
        DECRIPTION IS {text_content_description}

        """
        generated_description_gemini = gemini_request(prompt_gemini,api_index=int(str(i)[-1]))
        
        if generated_description_gemini == '':
            print(generated_description_gemini)
            generated_description_gemini = re.sub(r'["\[,\]\\]', ' ', workshop_loc['workshop_tags'])
            
        value = subCategories_df.loc[subCategories_df['id'] == workshop_loc["subCategory_id"]]
        subCategory = value.iloc[0]['name']
        value1 = categories_df.loc[categories_df['id'] == value.iloc[0]['category_id']]
        Category = value1.iloc[0]['name']
        
        
        title_name = '" ' + '#YUG-W-' + str(workshop_loc['id'])  + workshop_loc['title'].upper()+ ' "'
        full_text_workshops +=  "TITLE of the workshop is " + title_name + " and "
        full_text_workshops +=  "description of the "+ title_name +" event is " + text_content_description.replace("\r\n",'').replace('\n',' ')  + "\n"
        full_text_workshops +=  "area of interests and skills and technology and domains are required for " + title_name + " workshop are " + generated_description_gemini.replace('\n',' ').replace('*','')  + " and "
        full_text_workshops +=  "CATEGORY is " + Category + " and "
        full_text_workshops +=  "SUB CATEGORY is " + subCategory + ' and '
        full_text_workshops +=  "register link or url for " + title_name + " workshop is " + "https://yugam.in/w/" +  workshop_loc['workshop_url'] + ' and '
        full_text_workshops +=  title_name + " event happening on " + str(workshop_loc['startTime']) + ' and '
        full_text_workshops +=  "ENTRY FEES OR COST OR REGISTRATION FEES FOR " + title_name + "  workshop is " + str(workshop_loc['common_paymentAmount']) + ' and '
        full_text_workshops +=  "workshops tags are " + re.sub(r'["\[,\]\\]', ' ', workshop_loc['workshop_tags']) + '\n'
        
        
    with open("data/Yugam24.txt",'r',encoding="utf-8") as file:
        yugam=file.read()
            
    all_text_content_faiss_db = full_text_events + '\n' + full_text_workshops + '\n' + yugam
    
    with open('data/eventsWorkshops.txt', 'w', encoding='utf-8') as file:
        file.write(all_text_content_faiss_db)
    
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=100, chunk_overlap=1)
    chunks = text_splitter.split_text(all_text_content_faiss_db)

    print('No of documents (chunks) ==> ',len(chunks))

    embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True},)
    
    print('vector_store started.....')
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("source_data/faiss_check")
    print('vector_store ended.....')

    
updateFaissDB()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            