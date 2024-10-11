import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def authenticate_google_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive

def fetch_drive_folders(drive):
    folder_list = []
    query = "mimeType='application/vnd.google-apps.folder'"
    file_list = drive.ListFile({'q': query}).GetList()

    for file in file_list:
        folder_list.append({
            'id': file['id'],
            'name': file['title']
        })
    
    return folder_list

def fetch_drive_files(drive, folder_id):
    file_list = []
    query = f"'{folder_id}' in parents"
    files = drive.ListFile({'q': query}).GetList()

    for file in files:
        file_list.append({
            'id': file['id'],
            'name': file['title'],
            'mimeType': file['mimeType']
        })

    return file_list

def load_documents_from_memory(files_content):
    all_docs = []
    
    for file in files_content:
        try:
            loader = GoogleDriveLoader(
                    folder_id=file['id'],
                    recursive=False
                    )  
            documents = loader.load() 
         
            if documents:  
                splitter = CharacterTextSplitter(chunk_size=650, chunk_overlap=50)
                docs = splitter.split_documents(documents)
                all_docs.extend(docs)
            else:
                st.warning(f"No content found in file: {file['name']}")
        
        except Exception as e:
            st.error(f"Error loading file {file['name']}: {str(e)}")
    
    embeddings = HuggingFaceBgeEmbeddings()
    index = FAISS.from_documents(all_docs, embeddings)
    
    return index


def load_local_documents(local_folder):
    all_docs = []
    
    for filename in os.listdir(local_folder):
        file_path = os.path.join(local_folder, filename)

        if os.path.isfile(file_path):  # Ensure it's a file and not a directory
            try:
                loader = TextLoader(file_path)
                documents = loader.load()

                if documents:
                    splitter = CharacterTextSplitter(chunk_size=650, chunk_overlap=50)
                    docs = splitter.split_documents(documents)
                    all_docs.extend(docs)
            except Exception as e:
                st.error(f"Error loading file {filename}: {str(e)}")
    
    embeddings = HuggingFaceBgeEmbeddings()
    index = FAISS.from_documents(all_docs, embeddings)
    
    return index


st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title('🦙💬 LLaMA2 Chatbot')
    st.write('This chatbot uses the local LLaMA2 model with Ollama for document querying and general conversation.')

    st.subheader('Model and parameters')
    st.success('Using local LLaMA2 model with Ollama.', icon='✅')

    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    top_p = st.sidebar.slider('Top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('Max_length', min_value=20, max_value=80, value=50, step=5)

    index = None
    drive = None

    if st.sidebar.button('Connect to Google Drive'):
        drive = authenticate_google_drive()
        folders = fetch_drive_folders(drive)

        if folders:
            folder_names = [folder['name'] for folder in folders]
            selected_folder = st.selectbox('Select a folder:', folder_names)

            if selected_folder:
                folder_id = next(folder['id'] for folder in folders if folder['name'] == selected_folder)
                files_content = fetch_drive_files(drive, folder_id)

                if files_content:
                    index = load_documents_from_memory(files_content)
                    st.success(f"Loaded documents from folder: {selected_folder}")
                else:
                    st.warning("No files found in the selected folder.")
        else:
            st.warning("No folders found in Google Drive.")
    
    if index is None:
        local_folders = [f for f in os.listdir() if os.path.isdir(f)]
        selected_local_folder = st.selectbox("Select a local folder:", local_folders)
        
        if selected_local_folder:
            index = load_local_documents(selected_local_folder)
            st.success(f"Loaded documents from local folder: {selected_local_folder}")

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

    # if "document" in prompt.lower() or "local" in prompt.lower(): 
    with st.chat_message("assistant"):
        with st.spinner(f"Searching through documents..."):
            response = query_local_docs(prompt)
            st.write(f"Answer: {response}")
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
    # else:
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             response = generate_llama2_response(prompt)
    #             placeholder = st.empty()
    #             full_response = ''
    #             for item in response:
    #                 full_response += item
    #                 placeholder.markdown(full_response)
    #             placeholder.markdown(full_response)
    #     message = {"role": "assistant", "content": full_response}
    #     st.session_state.messages.append(message)
