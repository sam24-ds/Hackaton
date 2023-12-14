from fastapi import FastAPI, HTTPException
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import json
import os
import qdrant_client
import openai
from utils import *
import time


load_dotenv()

QDRANT_HOST=os.getenv("QDRANT_HOST")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
#QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

OPENAI_DEPLOYMENT_NAME_gpt35 = os.getenv("OPENAI_DEPLOYMENT_NAME_gpt35")
OPENAI_MODEL_NAME_gpt35 = os.getenv("OPENAI_MODEL_NAME_gpt35")

OPENAI_DEPLOYMENT_NAME_gpt35_16k = os.getenv("OPENAI_DEPLOYMENT_NAME_gpt35_16k")
OPENAI_MODEL_NAME_gpt35_16k = os.getenv("OPENAI_MODEL_NAME_gpt35_16k")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

# Configurations FastAPI
app = FastAPI()

# Configurations QdrantClient
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)


'''
def find_match(input_query, collect_name):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_DEPLOYMENT_NAME, chunk_size=1)
    response = embeddings.embed_query(
         input_query
    )
    embedds=response   #['data'][0]['embedding']

    search_result = client.search(
        collection_name=collect_name,
        query_vector=embedds,
        limit=3
    )

    for result in search_result:
        text_match = result.payload['text']

    return text_match


def query_refiner(conversation, query):
    
    response = openai.Completion.create(
    engine=OPENAI_DEPLOYMENT_NAME_gpt35,
    prompt=f"À partir de la requête utilisateur et du journal de conversation suivants, formules une question qui serait la plus pertinente pour fournir à l'utilisateur une réponse à partir d'une base de connaissances.\n\nJOURNAL DE CONVERSATION: \n{conversation}\n\nRequête: {query}\ n\nRequête affinée:",
    temperature=0.3,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

'''

def create_collect(collect_name):
    
    try:
        #indiquer ici le nom de la collection à créer 
        #collect_name=""
        '''
        client = qdrant_client.QdrantClient(
                QDRANT_HOST,
                api_key=QDRANT_API_KEY
            )
        '''

        # creation d'une collection dans qdrant


        collection_config = qdrant_client.http.models.VectorParams(
                size=1536, # 768 pour instructor-xl, 1536 pour OpenAI
                distance=qdrant_client.http.models.Distance.COSINE
            )

        client.recreate_collection(
            collection_name=collect_name,
            vectors_config=collection_config
        )
        return {"message": "Création de collection réussi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de la création de la collection : {e}")


def ask_question_with_context(qa, query, chat_history):
    #query = "what is Azure OpenAI Service?"
    result = qa({"question": query, "chat_history": chat_history})
    #print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    #print(chat_history)
    return result, chat_history


def chat_with_bot(collect_name: str, query: str, json_data: str, model: Optional[str] = "gpt-35-turbo", temp: Optional[float] = 0.3,  templa: Optional[str] = "En tant qu'agent du service clientèle, vous devez fournir une réponse utile et professionnelle à la question ou au problème de l'utilisateur"):
    #llm = ChatOpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=temp)
     
    openai_deployment_name=""

    if model == "gpt-35-turbo":
        openai_deployment_name="gpt-35-turbo"
    elif model =="gpt-35-turbo-16k":
        openai_deployment_name="gpt-35-turbo-16k"


    llm = AzureChatOpenAI(deployment_name=openai_deployment_name,
                      model_name=model,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      temperature=temp
                      )
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
     templa+"""templa
       Réponds à la question de l'utilisateur en fonction du context:\n{context}.
       """
     )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
     )
    

    QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

    vectorstore = Qdrant(
                client=client,
                collection_name=collect_name,
                embeddings=embeddings
            )
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=vectorstore.as_retriever(search_kwargs={'k':2}),
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False,
                                            condense_question_llm=llm,
                                            combine_docs_chain_kwargs={
                                            "prompt": ChatPromptTemplate.from_messages([
                                               system_message_prompt,
                                               human_message_prompt,
                                                                   ])}
                                              )
    
    #bot_response = dbqa({'query':query})
    #print(bot_response['source_documents'])
    
    history = get_history(json_data)
    chat_history= history
    bot_response, chat_history = ask_question_with_context(qa, query, chat_history)
    print(chat_history)
    return bot_response


def create_data_link(url):
    text = read_link(url)
    chunks = get_text_chunks(text)
    return chunks

debut_fonction_creadatadoc = time.time()

def create_data_doc(directory):
    text = process_files_in_directory(directory)
    chunks = get_text_chunks(text)
    return chunks


'''
def get_embedd(text_chunks, model_id="text-embedding-ada-002"):
      embeddings = OpenAIEmbeddings(model=model_id, chunk_size=1)
      points = []
      for idx, chunk in enumerate(text_chunks):
            #print (chunk.page_content[:])
            response =embeddings.embed_query(
             
            str(chunk.page_content[:])
               
            )
            embedd = response #['data'][0]['embedding']
               
            point_id = str(uuid.uuid4())
            points.append(qdrant_client.http.models.PointStruct(id=point_id, vector=embedd, payload={"text":chunk}))

      return points
'''


def insert_data_doc(collect_name, directory):
    
    try:
        
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

        vectorstore = Qdrant(
                client=client,
                collection_name=collect_name,
                embeddings=embeddings
            )
        chunks = create_data_doc(directory)

        if len(chunks)>= 110000:
            chunk_size = 110000
            num_chunks = len(chunks)
            num_batches = (num_chunks + chunk_size - 1) // chunk_size

            for i in range(num_batches):
              start_idx = i * chunk_size
              end_idx = min((i + 1) * chunk_size, num_chunks)
              batch = chunks[start_idx:end_idx]
            
              vectorstore.add_documents(batch)
        else:
            
            vectorstore.add_documents(chunks)
        
        return {"message": "Insertion de la données réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de l'insertion des textes : {e}")
    

def insert_data_link(collect_name, url):
    
    try:
        texts=[]
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

        vectorstore = Qdrant(
                client=client,
                collection_name=collect_name,
                embeddings=embeddings
            )
        chunks = create_data_link(url)

        
        
        vectorstore.add_documents(chunks)
        
        return {"message": "Insertion de la données réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de l'indexation des textes : {e}")



def delete_collect(collect_name):
    try:
        client.delete_collection(collection_name=collect_name)
        return {"message": f"La collection {collect_name} a été supprimée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {e}")


def get_info_collection(collect_name):
    try:
        info = client.get_collection(collection_name=collect_name)
        #print("collection info:", info)
        return {"collection_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {e}")


# Créer la route pour la fonction create_collect
@app.post("/create_collection/{collect_name}", response_model=dict)
def create_collection_route(collect_name: str):
    return create_collect(collect_name)

# Créer la route pour la fonction insert_data
@app.post("/insert_data/{collect_name}/{directory}/{source_type}", response_model=dict)
def insert_data_route(collect_name: str, source_type: str, directory_or_link: str):
    if source_type.lower() == "directory":
        return insert_data_doc(collect_name, directory_or_link)
    
    elif source_type.lower() == "link":
        # Traiter les données à partir du lien web
        return insert_data_link(collect_name,directory_or_link)
    else:
        raise ValueError("Le type de source doit être 'directory' ou 'link'.")


@app.delete("/delete-collection")
def delete_collection(collect_name: str):
    return delete_collect(collect_name)


@app.get("/collection-info")
def collection_info(collect_name: str):
    return get_info_collection(collect_name)


@app.post("/chatbot/")
def chatbot_interaction(collect_name: str, query: str, json_data: str, model: Optional[str] = "gpt-35-turbo", temp: Optional[float] = 0.3,  templa: Optional[str] = "En tant qu'agent du service clientèle, vous devez fournir une réponse utile et professionnelle à la question ou au problème de l'utilisateur."):
    bot_response = chat_with_bot(collect_name, query,json_data, model, temp, templa)
    return {"bot_response": bot_response}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
