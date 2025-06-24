import os
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

# Constants
BUCKET_NAME = os.getenv("BUCKET_NAME")
FOLDER_PATH = "/tmp"
FAISS_INDEX_NAME = "my_faiss"
MODEL_ID = "meta.llama3-8b-instruct-v1:0"

# AWS Clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Embedding model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Generate unique identifier
def get_unique_id() -> str:
    return str(uuid.uuid4())

# Download FAISS index files from S3
def load_index_from_s3():
    s3_client.download_file(BUCKET_NAME, f"{FAISS_INDEX_NAME}.faiss", f"{FOLDER_PATH}/{FAISS_INDEX_NAME}.faiss")
    s3_client.download_file(BUCKET_NAME, f"{FAISS_INDEX_NAME}.pkl", f"{FOLDER_PATH}/{FAISS_INDEX_NAME}.pkl")

# Load Bedrock LLM (Meta LLaMA 3)
def get_llm():
    return Bedrock(
        model_id=MODEL_ID,
        client=bedrock_client,
        model_kwargs={
            "max_gen_len": 512,
            "temperature": 0.7
        }
    )

# Perform retrieval-augmented generation
def get_response(llm, vectorstore, question: str) -> str:
    template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain({"query": question})
    return result["result"]

# Streamlit App
def main():
    st.header("📄 Chat with PDF using Amazon Bedrock + RAG")

    # Load FAISS vector index
    load_index_from_s3()
    st.success("✅ Vector store loaded from S3.")

    # Show current temp directory contents
    st.write(f"📁 Files in `{FOLDER_PATH}`:")
    st.write(os.listdir(FOLDER_PATH))

    # Load FAISS index locally
    faiss_index = FAISS.load_local(
        index_name=FAISS_INDEX_NAME,
        folder_path=FOLDER_PATH,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.success("✅ FAISS index loaded and ready.")

    # User Input
    question = st.text_input("❓ Ask your question based on the PDF content:")

    if st.button("🔍 Ask Question") and question:
        with st.spinner("🧠 Generating answer..."):
            llm = get_llm()
            response = get_response(llm, faiss_index, question)
            st.write("💬 **Answer:**")
            st.write(response)
            st.success("✅ Done.")

if __name__ == "__main__":
    main()
