from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

#from langchain import PromptTemplate
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os
import qdrant_client
import openai
from utils import *
import uuid
import time
import tiktoken
import tempfile


load_dotenv()

QDRANT_HOST=os.getenv("QDRANT_HOST")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY

# Configurations FastAPI
app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configurations QdrantClient
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
    timeout=60.0
)



def find_match(input_query, collect_name, user_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    response = embeddings.embed_query(
         input_query
    )
    embedds=response   #['data'][0]['embedding']

    search_result = client.search(
        collection_name=collect_name,

        query_filter=qdrant_client.http.models.Filter(
        must=[
            qdrant_client.http.models.FieldCondition(
                key="group_id",
                match=qdrant_client.http.models.MatchValue(
                    value=user_id,
                ),
            )
        ]
    ),

        query_vector=embedds,
        limit=3
    )
    tt= []
    for result in search_result:
        #print(result)
        tt.append(result)
        
    if len(tt) > 1:
        text_match = tt[0].payload['text']+tt[1].payload['text']
    elif len(tt)==1:
        text_match = tt[0].payload['text']
    else:
        text_match=""
    #print(tt[0].payload['text']['page_content'])
    return text_match


def query_refiner(conversation, query):
    system_prompt = "je veux que tu agis en tant qu'expert en rédaction de question"
    prompt=f"""À partir de la question de l'utilisateur et de l'historique de la conversation suivants, formules une question qui serait la plus pertinente possible pour fournir à l'utilisateur une réponse à partir d'une base de connaissances.\n\nHISTORIQUE DE CONVERSATION: \n{conversation}\n\nQuestion: {query}\ n\nRequête affinée:"""
    response = openai.ChatCompletion.create(
    model= "gpt-3.5-turbo",
    
    temperature=0.3,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    #tt=response['choices'][0]["text"]
    #print(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]




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
            vectors_config=collection_config,
            hnsw_config=qdrant_client.http.models.HnswConfigDiff(
             payload_m=16,
                      m=0),
            optimizers_config=qdrant_client.http.models.OptimizersConfigDiff(memmap_threshold=20000)          
        )
        return {"message": "Création de collection réussi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de la création de la collection : {e}")
    

def create_index_payload(collection_name, group_id):

    try:
        client.create_payload_index(
        collection_name=collection_name, 
        field_name= group_id, 
        field_schema=qdrant_client.http.models.PayloadSchemaType.KEYWORD
        )
        return {"message": "Création de l'index personnalisé réussi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de la création de l'indexe personnalisé : {e}")


def ask_question_with_context(qa, query, chat_history):
    #query = "what is Azure OpenAI Service?"
    result = qa({"question": query, "chat_history": chat_history})
    #print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    #print(chat_history)
    return result, chat_history


def chat_with_bot(collect_name: str, user_id: str, query: str, json_data: str, model: Optional[str] = "gpt-3.5-turbo", temp: Optional[float] = 0.1,  templa: Optional[str] = "En tant qu'agent du service clientèle, vous devez fournir une réponse utile et professionnelle à la question ou au problème de l'utilisateur."):
    #llm = ChatOpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=temp)
     
    

    llm = ChatOpenAI(
                      model_name=model,
                      openai_api_key=OPENAI_API_KEY,
                      temperature=temp,
                      
                      )
    


    #rr = "Renvoyez votre réponse en espagnol"
    system_msg_template = SystemMessagePromptTemplate.from_template(template=f"""Répondez en Français et de manière conviviale à la question de l'utilisateur en exploitant au maximum les informations contenu dans le contexte fourni tout en respectant toujours le style suivant {templa}, et si la réponse n'est pas contenue dans le contexte, dites uniquement 'je ne sais pas'.""")


    human_msg_template = HumanMessagePromptTemplate.from_template(template="""{input}""")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template,MessagesPlaceholder(variable_name="history"),human_msg_template])

    memory_v = ConversationBufferWindowMemory(k=3, return_messages=True)
    conversation_string = get_conversation(json_data)  # Utilisez la fonction appropriée pour récupérer la conversation
    refined_query = query_refiner(conversation_string, query)
    #print(conversation_string )

    #print(refined_query)
    context = find_match(refined_query, collect_name, user_id)
    #print(context)
    
    conversation = ConversationChain(memory=memory_v ,prompt=prompt_template, llm=llm, verbose=False)
    bot_response = conversation.predict(input=f"Contexte:\n {context} \n\n Question:\n{query}")
    #print(conversation_string )
    

    return bot_response


def create_data_link(url):
    text = read_link(url)
    chunks = get_text_chunks(text)
    return chunks


def create_data_doc(directory):
    text = process_files_in_directory(directory)
    chunks = get_text_chunks(text)
    return chunks



def get_embedd(text_chunks,  model_id, user_id):
      embeddings = OpenAIEmbeddings(model=model_id, chunk_size=1)
      points = []
      for idx, chunk in enumerate(text_chunks):
            print (type(chunk.page_content[:]))
            response =embeddings.embed_documents(
             
            [chunk.page_content[:]]
               
            )
            embedd = response[0]
               
            point_id = str(uuid.uuid4())
            points.append(qdrant_client.http.models.PointStruct(id=point_id, payload={"text": chunk, "group_id":user_id},  vector=embedd))

      return points


def get_embeding(chunks, model_id, user_id):
    points = []
    for idx, chunk in enumerate(chunks):
        response = openai.Embedding.create(
            input=chunk.page_content[:],
            model=model_id
            
        )
        embeddings = response['data'][0]['embedding']
        points_id = str(uuid.uuid4())
        points.append(qdrant_client.http.models.PointStruct(id=points_id, vector=embeddings, payload={"text": chunk.page_content[:], "group_id":user_id}))
    return points

def insert_data(get_points, collect_name):
    operation_info = client.upsert(
        collection_name=collect_name,
        wait=True,
        points=get_points
    )


def insert_data_doc(collect_name,user_id, directory):
    
    try:
        
        
        chunks = create_data_doc(directory)

        points= get_embeding(chunks, "text-embedding-ada-002", user_id)
        insert_data(points, collect_name)
        
        return {"message": "Insertion de la données réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de l'insertion des textes : {e}")
    

def insert_data_link(collect_name, user_id, url ):
    
    try:
        
        
        chunks = create_data_link(url)

        points=get_embeding(chunks, "text-embedding-ada-002", user_id)
        insert_data(points, collect_name)
        
        return {"message": "Insertion de la données réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors de l'insertion des textes : {e}")


def delete_collect(collect_name):
    try:
        client.delete_collection(collection_name=collect_name)
        return {"message": f"La collection {collect_name} a été supprimée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {e}")
    
def delete_points(collect_name, user_id):
        
    try:
        client.delete(
        collection_name=collect_name,
        points_selector=qdrant_client.http.models.FilterSelector(
        filter=qdrant_client.http.models.Filter(
            must=[
                qdrant_client.http.models.FieldCondition(
                    key="group_id",
                    match=qdrant_client.http.models.MatchValue(value=user_id),
                        ),
                    ],
                )
              ),
            )
        return {"message": f"Les ou le vecteur avec l'id {user_id} de la collection {collect_name} a été supprimée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {e}")

def get_info_collection(collect_name):
    try:
        info = client.get_collection(collection_name=collect_name)
        #print("collection info:", info)
        return {"collection_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {e}")
    
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
    
def summarize(text):
    system_prompt = "je veux que tu agis en tant qu'expert en rédaction de résumé de document pédagogique"
    prompt = f'''créer le résumé du texte suivant:
    Text:{text}
    Ajoute un titre au résumé.
    Ton résumé doit être informatif et factuel, couvrant les aspects les plus importants du sujet.
    Commence ton résumé avec un PARAGRAPHE D'INTRODUCTION qui donne un aperçu du sujet suivi de PUCES informatifs si possible ET termine le résumé par une CONCLUSION'''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0
    )
    r = response["choices"][0]["message"]["content"]
    return r


def summarize_final(transcription):
    num_tokens = num_tokens_from_string(transcription, "cl100k_base")
    if num_tokens > 15000:
        result = 'Votre document est trop long'
    else:
        result = summarize(transcription)
    return result


# Créer la route pour la fonction create_collect
@app.post("/create_collection/{collect_name}", response_model=dict)
def create_collection_route(collect_name: str):
    return create_collect(collect_name)

#Créer la route pour la fonction create_index_payload
@app.post("/create_index_payload/{collect_name}/{groupe_id}", response_model=dict)
def create_index_payload_route(collect_name: str, groupe_id: str):
    return create_index_payload(collect_name, groupe_id)


# Créer la route pour la fonction insert_data
@app.post("/insert_data/{collect_name}/{user_id}/{source_type}", response_model=dict)
def insert_data_route(collect_name: str, user_id: str, source_type: str, directory_or_link: str):
    if source_type.lower() == "directory":
        return insert_data_doc(collect_name, user_id, directory_or_link)
    
    elif source_type.lower() == "link":
        # Traiter les données à partir du lien web
        return insert_data_link(collect_name, user_id,directory_or_link)
    else:
        raise ValueError("Le type de source doit être 'directory' ou 'link'.")


@app.delete("/delete-collection")
def delete_collection(collect_name: str):
    return delete_collect(collect_name)


@app.delete("/delete-vectors")
def delete_vectors(collect_name: str, user_id: str):
    return delete_points(collect_name, user_id)


@app.get("/collection-info")
def collection_info(collect_name: str):
    return get_info_collection(collect_name)


@app.post("/chatbot/")
def chatbot_interaction(collect_name: str, user_id: str, query: str, json_data: str, model: Optional[str] = "gpt-3.5-turbo", temp: Optional[float] = 0.3,  templa: Optional[str] = "En tant qu'assistant scolaire , vous devez fournir une réponse utile et informatif à la question ou au problème de l'utilisateur en utilisant un ton conviviale et pédagogie."):
    bot_response = chat_with_bot(collect_name, user_id, query, json_data, model, temp, templa)
    return {"bot_response": bot_response}



    


@app.post("/summarize-2/")
async def transcribe_summarize(file: UploadFile = File(...)):
   # Write the uploaded file to a temporary file
   with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name

   # Now you can pass temp_path to read_doc
   doc = read_doc(temp_path)
   txt = doc[0].page_content[:]
   bot_response = summarize_final(txt)

   # Don't forget to delete the temporary file when you're done with it
   os.unlink(temp_path)

   return {"bot_response": bot_response}







if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
