from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
import tiktoken
import openai
import os
from dotenv import load_dotenv
import json
from pytesseract import image_to_string



def read_pdf(file):
    loader = PyPDFLoader(file) 
    doc = loader.load()
    return doc


def read_doc(file_path):
    loader = UnstructuredFileLoader(file_path)
    doc = loader.load()
    #content = "".join([str(el.page_content[:]) for el in doc])
    return doc

def read_link(url):
    urls = [url]
    loader = SeleniumURLLoader(urls=urls)
    doc = loader.load()

    return doc


def process_file(file_path):
    if file_path.lower().endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".jpg") or file_path.endswith(".png"):
        return  image_to_string(file_path)
    else :
        return read_doc(file_path)
    

def process_files_in_directory(directory):
    documents_list = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            document = process_file(file_path)
            documents_list.extend(document)
    return documents_list






def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


      
def get_conversation_string(history):
    conversation_string = ""

    # Convertir la liste en une seule chaîne de caractères
    history_str = '\n'.join(history)

    # Diviser les messages du chatbot en une liste de tuples (user, bot)
    messages = history_str.split('\nAI: ')
    if len(messages) > 1:
        messages = [('User: ' + messages[i].replace('Human: ', '').strip(), 'AI: ' + messages[i+1].strip()) for i in range(0, len(messages)-1, 2)]

        for user_message, bot_message in messages:
            conversation_string += user_message + "\n"
            conversation_string += bot_message + "\n"

    return conversation_string.strip()

def get_history(json_data):
    
    data = json.loads(json_data)

    # Créer une liste de tuples à partir des données JSON
    result = [(item["query"], item["response"]) for item in data]

    return result
# get_history--> format attendu en entré :[{"query":"qui est Samir BONI ?", "response":"Samir BONI est une personne dont les informations personnelles et professionnelles sont présentées dans le texte que vous avez fourni. D'après les informations fournies, il est de nationalité béninoise, compétent en informatique et possède des compétences en programmation Python3, SQL, analyse de données, machine learning, deep learning, NLP et analyses UML. Il a également effectué plusieurs stages professionnels et académiques dans des entreprises et organisations telles que OGI, MDB BENIN et Digit Consulting, et travaille actuellement comme développeur en intelligence artificielle chez Digit consult."}]


def get_conversation(json_data):
    data = json.loads(json_data)

    if not data:
        return ""

    # Construire la chaîne de caractères formatée
    formatted_string = ""
    for item in data:
        formatted_string += f"User: {item['User']}\nBot: {item['Bot']}\n"
    
    return formatted_string











