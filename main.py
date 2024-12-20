import streamlit as st
import tempfile
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import OpenAI


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\x00', '', text)  
    text = re.sub(r'\xa0', ' ', text)  
    return text


def create_vectordb_from_tempfile(pdf_tempfile_path, db_name):
    loader = PyPDFLoader(pdf_tempfile_path)
    pages = loader.load_and_split()
    st.write('Total Pages:',len(pages))
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    docs = text_splitter.split_documents(pages)
    st.write('Total documents:',len(docs))
    cleaned_docs = [Document(page_content=clean_text(d.page_content), metadata=d.metadata) for d in docs]
    st.write('Total documents after cleaning:',len(cleaned_docs))
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_directory = os.path.join("vector_databases", db_name)
    vectordb = Chroma.from_documents(cleaned_docs, embedding_function, persist_directory=persist_directory)  

    return vectordb


def load_vector(db_name):
    persist_directory = os.path.join("vector_databases", db_name)

    if os.path.exists(persist_directory):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
        return vectordb
    else:
        return None


if "vectordb" in st.session_state:
    del st.session_state.vectordb  

st.title("Chat with your PDF")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file is not None:
    with st.spinner("Processing your PDF, please wait..."):
        pdf_filename = os.path.splitext(pdf_file.name)[0]  

        
        vectordb = load_vector(pdf_filename)

        if vectordb is None:  
            if not os.path.exists("vector_databases"):
                os.makedirs("vector_databases")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(pdf_file.read())
                tmpfile_path = tmpfile.name

            vectordb = create_vectordb_from_tempfile(tmpfile_path, pdf_filename)

        st.session_state.vectordb = vectordb  # Save the vectordb to session state


# Ensure the vectorDB is loaded before asking questions
if "vectordb" in st.session_state:
    vectordb = st.session_state.vectordb
    llm = OpenAI(api_key=os.getenv("api_key"),temperature=0.2)

    prompt_template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    question = st.text_input("Ask your question:")

    if question:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        chain = (
            {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        result = chain.invoke( question)

        st.write("Answer:", result)

        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.append((question, result))
        
        if st.checkbox("Show Conversation History"):
            for q, a in st.session_state.conversation_history:
                st.write("Q:", q)
                st.write("A:", a)
