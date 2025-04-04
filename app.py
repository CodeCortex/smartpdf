import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import GoogleGenerativeAI

from htmlTemplates import css, bot_template, user_template

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# Create vectorstore using HuggingFace embeddings
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Create conversational retrieval chain
def get_conversation_chain(vstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)
    
   

def main():
    st.set_page_config(page_title="Al-Powered PDF Reader with LangChain", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("Al-Powered PDF Reader with LangChain 📚")

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
   
    st.write(user_template.replace("{{MSG}}", "Hello robot 👋"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human 🤖"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                # Get text
                text = get_pdf_text(pdf_docs)

                # Get chunks
                chunks = get_text_chunks(text)
                st.success("✅ Text Chunking Done")

                # Vector store
                vectorstore = get_vectorstore(chunks)
                st.success("✅ Vector Store Created")

                # Conversation chain
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Chat is Ready!")

  

if __name__ == "__main__":
    main()
