import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize Streamlit app
st.title("Citizen Complaint Assistant")

# Path to the CSV file
csv_file_path = 'emails.csv'

if os.path.exists(csv_file_path):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Check if 'email' column exists
    if 'email' not in df.columns:
        st.error("CSV file must contain an 'email' column.")
    else:
        # Function to extract metadata from each email
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

        # Create Document objects from emails with metadata
        documents = [
            Document(page_content=email, metadata=extract_metadata(email))
            for email in df['email']
        ]

        # Split documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = DocArrayInMemorySearch.from_documents(split_docs, embeddings)

        # Initialize OpenAI Chat LLM
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')  # Choose a stable model

        # Create RetrievalQA chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 40})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User input for querying the system
        query = st.text_input("Ask a question about the complaints:")

        if query:
            with st.spinner("Generating response..."):
                response = qa_chain.run(query)
            st.write("### Response")
            st.write(response)
else:
    st.error(f"CSV file '{csv_file_path}' not found in the project directory.")
