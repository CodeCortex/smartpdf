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

def convert_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def convert_text_to_textChunk(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len
    )
    return splitter.split_text(text)

# Create vectorstore using HuggingFace embeddings
def convert_chunks_to_vector_store(chunks):
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


def summarize_text(text):
    prompt = f"""
    You are a helpful assistant. Please summarize the following document in a short paragraph followed by bullet points that highlight the key points:

    --- DOCUMENT START ---
    {text}
    --- DOCUMENT END ---

    Summary:
    """
    response = llm.invoke(prompt)

    lines = response.split("\n")
    
    # First line(s) could be empty, find the first non-empty line as summary paragraph
    summary_paragraph = ""
    bullet_points = []
    
    for line in lines:
        line = line.strip()
        if line:
            if summary_paragraph == "":
                summary_paragraph = line
            else:
                bullet_points.append(line)
    
    formatted_bullet_points = "\n".join(bullet_points)
    
    formatted_summary = f"{summary_paragraph}\n\n{formatted_bullet_points}"
    
    return formatted_summary



def handle_userinput(user_question):
    # Safely call the conversation object from session_state
    if st.session_state.conversation is None or st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDF files first.", icon="⚠️")

        return
    
    response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history = response['chat_history']
    
    # Reverse the chat history in user-bot pairs
    chat_history = st.session_state.chat_history
    pairs = [chat_history[i:i+2] for i in range(0, len(chat_history), 2)]
    reversed_pairs = reversed(pairs)

    for pair in reversed_pairs:
        for message in pair:
            if hasattr(message, "type") and message.type == "human":
                st.write(user_template.replace("{{MESSAGE}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MESSAGE}}", message.content), unsafe_allow_html=True)

   

def main():
    st.set_page_config(page_title="Al-Powered PDF Reader with LangChain", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("Al-Powered PDF Reader with LangChain 📚")

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "summary" not in st.session_state:
        st.session_state.summary= None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
        
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                # Get text
                text = convert_pdf_text(pdf_docs)

                # Get chunks
                chunks = convert_text_to_textChunk(text)
                st.success("✅ Text Chunking Done")

                # Vector store
                vectorstore = convert_chunks_to_vector_store(chunks)
                st.success("✅ Vector Store Created")

                # Conversation chain
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Chat is Ready!")
                
        if st.button("Summarize") and pdf_docs:
            with st.spinner("Summarizing..."):
                text= convert_pdf_text(pdf_docs)
                summary= summarize_text(text)
                st.session_state.summary= summary
                st.success("✅ Summary generated!")
                
        if st.session_state.summary:
            st.markdown(f"📄 **Summary:**\n\n{st.session_state.summary}")
                
        
        


        
        # Placeholder for MCP Server / Playwright usage
        st.markdown("---")
        st.info("🚀 Playwright MCP automation coming soon...")


  

if __name__ == "__main__":
    main()
