import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(docs, user_question):
    prompt_template = """
    You are a book chatbot that uses information from the book data in the context and provide useful answers to the user.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say what the context is talking about briefly, don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.3)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"context": docs, "question": user_question})
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    #
    # return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    answer = get_conversational_chain(docs, user_question)
    return answer


def extract_text_from_documents(documents):
    text = ""
    for doc in documents:
        text += doc.page_content + "\n"
    return text

def io(question):
    if not os.path.isdir('faiss_index'):
        file_path = "Atomic Habits by James Clear (PDF) PDFDrive ( PDFDrive ).pdf"
        loader = PDFPlumberLoader(file_path)
        docs = loader.load_and_split()
        get_vector_store(get_text_chunks(extract_text_from_documents(docs)))
    #user_question = input("User: \n")
    if question:
        return user_input(question)




    # raw_text = all_text
    # text_chunks = get_text_chunks(raw_text)
    # get_vector_store(text_chunks)

    # if user_question:
    #     user_input(user_question)
