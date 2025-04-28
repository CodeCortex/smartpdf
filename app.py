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

import collections
import re

load_dotenv()

# Function to create LLM dynamically
def create_llm(model_name, temperature):
    return GoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def extract_keywords(text, num_keywords=5):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = collections.Counter(words).most_common(num_keywords)
    return [word for word, count in common_words]


def convert_pdf_text(pdf_docs):
    all_text = {}
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        all_text[pdf.name] = text
    return all_text


def convert_text_to_textChunk(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len
    )
    return splitter.split_text(text)

# Create vectorstore
def convert_chunks_to_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Conversation Chain
def get_conversation_chain(vstore, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# Summarize each document separately
def summarize_each_text(text_dict, llm):
    summaries = {}
    for filename, text in text_dict.items():
        prompt = f"""
        You are a helpful assistant. Summarize the following document briefly and list the key points:

        --- DOCUMENT START ---
        {text}
        --- DOCUMENT END ---

        Summary:
        """
        response = llm.invoke(prompt)
        
        summaries[filename] = response.strip()
    return summaries

def handle_userinput(user_question):
    if st.session_state.conversation is None or st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDF files first.", icon="⚠️")
        return
    
    response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history = response['chat_history']
    
    # Reverse the chat history
    chat_history = st.session_state.chat_history
    pairs = [chat_history[i:i+2] for i in range(0, len(chat_history), 2)]
    reversed_pairs = reversed(pairs)

    for pair in reversed_pairs:
        for message in pair:
            if hasattr(message, "type") and message.type == "human":
                st.write(user_template.replace("{{MESSAGE}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MESSAGE}}", message.content), unsafe_allow_html=True)


def download_chat_history():
    if st.session_state.chat_history:
        history_text = ""
        for message in st.session_state.chat_history:
            speaker = "User" if message.type == "human" else "Bot"
            history_text += f"{speaker}: {message.content}\n\n"
        st.download_button("Download Chat History", history_text, file_name="chat_history.txt")


def main():
    st.set_page_config(page_title="AI PDF Reader with LangChain", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("AI-Powered PDF Reader with LangChain 📚")

    # Session States
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
    if "text_data" not in st.session_state:
        st.session_state.text_data = None
        
    if "summaries" not in st.session_state:
        st.session_state.summaries = None
        
    if "llm" not in st.session_state:
        st.session_state.llm = create_llm(model_name="gemini-2.0-flash", temperature=0.7)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Settings")
        temp = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7)
        model_choice = st.selectbox("Choose Model", ["gemini-2.0-flash", "gemini-1.5-pro"])

        if st.button("Apply Settings"):
            st.session_state.llm = create_llm(model_choice, temp)
            st.success("✅ Settings Applied!")

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process PDFs") and pdf_docs:
            with st.spinner("Processing..."):
                text_data = convert_pdf_text(pdf_docs)
                combined_text = "\n".join(text_data.values())
                chunks = convert_text_to_textChunk(combined_text)
                vectorstore = convert_chunks_to_vector_store(chunks)

                st.session_state.text_data = text_data
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.llm)
                
                st.success("✅ PDFs processed and chat is ready!")
                
                # Display document insights
                st.subheader("📈 Document Insights")
                for filename, text in text_data.items():
                    word_count = len(text.split())
                    keyword_list = extract_keywords(text)
                    reader = PdfReader(next(pdf for pdf in pdf_docs if pdf.name == filename))
                    page_count = len(reader.pages)
                    
                    st.markdown(f"**{filename}**")
                    st.markdown(f"- Pages: {page_count}")
                    st.markdown(f"- Words: {word_count}")
                    st.markdown(f"- Top Keywords: {', '.join(keyword_list)}")

        if st.button("Summarize PDFs") and st.session_state.text_data:
            with st.spinner("Summarizing..."):
                summaries = summarize_each_text(st.session_state.text_data, st.session_state.llm)
                st.session_state.summaries = summaries
                st.success("✅ Summaries generated!")


        if st.session_state.summaries:
            st.subheader("📄 Document Summaries")
            for filename, summary in st.session_state.summaries.items():
                st.markdown(f"### {filename}")
                st.markdown(summary)


        st.markdown("---")
        st.subheader("Conversation")
        download_chat_history()


        st.markdown("---")
        st.info("🚀 Playwright MCP automation coming soon...")

if __name__ == "__main__":
    main()


