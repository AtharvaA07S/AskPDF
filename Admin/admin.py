import os
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
# ==== AWS & Langchain Setup ====

# S3 client and bucket
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("Environment variable 'BUCKET_NAME' is not set.")

# Bedrock embeddings
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# ==== Helper Functions ====

def get_unique_id() -> str:
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(pages)

def create_vector_store(request_id, documents):
    vector_store = FAISS.from_documents(documents, bedrock_embeddings)
    
    folder_path = "/tmp"
    file_name = f"{request_id}.bin"

    vector_store.save_local(index_name=file_name, folder_path=folder_path)

    # Upload FAISS files to S3
    s3_client.upload_file(
        Filename=f"{folder_path}/{file_name}.faiss",
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss"
    )
    s3_client.upload_file(
        Filename=f"{folder_path}/{file_name}.pkl",
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl"
    )

    return True

# ==== Main Streamlit App ====

def main():
    st.title("Admin: Chat with PDF (Demo)")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        request_id = get_unique_id()
        st.write(f"Request ID: {request_id}")

        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Total Pages: {len(pages)}")

        # Split text
        split_docs = split_text(pages)
        st.write(f"Chunks created: {len(split_docs)}")

        st.markdown("---")
        st.subheader("Sample Chunks")
        st.write(split_docs[0])
        if len(split_docs) > 1:
            st.write(split_docs[1])
        st.markdown("---")

        st.write("Creating vector store and uploading to S3...")
        if create_vector_store(request_id, split_docs):
            st.success("✅ PDF processed and uploaded to S3 successfully!")
        else:
            st.error("❌ Something went wrong during processing.")

# ==== Entry Point ====
if __name__ == "__main__":
    main()
