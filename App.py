import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

st.title("Citizen Complaint Assistant")

csv_file_path = 'emails.csv'

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'email' not in df.columns:
        st.error("CSV file must contain an 'email' column.")
    else:
        def extract_metadata(email_text):
            metadata = {}
            lines = email_text.split('\n')
            for i, line in enumerate(lines):
                if 'Location of Visual Pollution' in line:
                    metadata['location'] = lines[i + 1].strip() if i + 1 < len(lines) else ''
                elif 'Type and Nature of Visual Pollution' in line:
                    metadata['type'] = lines[i + 1].strip() if i + 1 < len(lines) else ''
                elif 'Severity or Impact of the Visual Pollution' in line:
                    metadata['severity'] = lines[i + 1].strip() if i + 1 < len(lines) else ''
                elif 'Date and Time of Complaint' in line:
                    metadata['date'] = lines[i + 1].strip() if i + 1 < len(lines) else ''
                elif 'Contact Information of the Complainant' in line:
                    contact_info = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip():
                        contact_info.append(lines[j].strip())
                        j += 1
                    metadata['contact'] = ' '.join(contact_info)
            return metadata

        documents = [
            Document(page_content=email, metadata=extract_metadata(email))
            for email in df['email']
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        # Use Chroma instead of FAISS
        vector_store = Chroma.from_documents(split_docs, embeddings)

        llm = ChatOpenAI(model_name='gpt-4o')  # Use your desired model
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 40})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask a question about the complaints:")

        if query:
            with st.spinner("Generating response..."):
                response = qa_chain.run(query)
            st.write("### Response")
            st.write(response)
else:
    st.error(f"CSV file '{csv_file_path}' not found in the project directory.")
