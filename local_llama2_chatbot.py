import streamlit as st
import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA 
from langchain_community.llms import Ollama


def load_documents(folder_path):
    all_docs = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif filename.endswith('.csv'):
            loader = CSVLoader(file_path)
            documents = loader.load()  
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            st.warning(f"Unsupported file type: {filename}")
            continue

        splitter = CharacterTextSplitter(chunk_size=650, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        all_docs.extend(docs)

    embeddings = HuggingFaceBgeEmbeddings()
    index = FAISS.from_documents(all_docs, embeddings)
    
    return index


folder_path = "docs/" 
index = load_documents(folder_path)


st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title('🦙💬 LLaMA2 Local Chatbot')
    st.write('This chatbot uses the local LLaMA2 model with Ollama for document querying and general conversation.')

    st.subheader('Model and parameters')
    st.success('Using local LLaMA2 model with Ollama.', icon='✅')

    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    top_p = st.sidebar.slider('Top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('Max_length', min_value=20, max_value=80, value=50, step=5)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


llm = Ollama(model='llama2')
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=index.as_retriever())

def query_local_docs(query):
    return qa_chain.run(query)

def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. Please respond with helpful and concise answers."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    output = llm(prompt_input, temperature=temperature, top_p=top_p, max_length=max_length)
    return output


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if "document" in prompt.lower() or "local" in prompt.lower(): 
        with st.chat_message("assistant"):
            with st.spinner(f"Searching through local documents in {folder_path} folder..."):
                response = query_local_docs(prompt)
                st.write(f"Answer: {response}")
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
    else:
        # Handle general chatbot queries with LLaMA2
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
