import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_csv_data(csv_files):
    all_data = []  # List to hold all data rows with column names
    
    for csv_file in csv_files:
        csv_text = csv_file.read().decode("utf-8")
        csv_reader = csv.reader(csv_text.splitlines())
        
        headers = next(csv_reader)  # Extracting the header row
        for row in csv_reader:
            if row:  # Ensuring the row is not empty
                row_data = {headers[i]: row[i] for i in range(len(row))}  # Create a dict with column names
                all_data.append(row_data)

    return all_data  # Return the list of dictionaries

def get_text_chunks(data):
    text = ""
    # Convert list of dictionaries to a structured string
    for entry in data:
        formatted_entry = ' '.join(f"{key}: {value}" for key, value in entry.items())
        text += formatted_entry + "\n"  # Separate entries with triple newlines

    splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are a CSV data assistant. Your task is to assist users in retrieving information from the provided CSV file. Your goal is to provide accurate answers based on the context of the data. If the answer is not available in the provided context, you should indicate that to the user.
    Use the provided context and question to generate an accurate response. If you encounter any issues or uncertainties, feel free to seek clarification or provide suggestions for refining the query.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.1,
                                   top_k=10,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some csv's and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response


def main():
    st.set_page_config(
        page_title="Gemini CSV Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading CSV files
    with st.sidebar:
        st.title("Menu:")
        csv_docs = st.file_uploader(
            "Upload your CSV Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if csv_docs is not None:
                with st.spinner("Processing..."):
                    raw_text = read_csv_data(csv_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:  # Check if text_chunks is not empty
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.warning("No data found in CSV files.")
            else:
                st.warning("Please upload a CSV file.")
                return  # Return to avoid further execution

    # Main content area for displaying chat messages
    st.title("Chat with CSV files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some CSVs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
    

if __name__ == "__main__":
    main()
