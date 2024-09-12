from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import tempfile

class chatpdf:


    
    def qa_pdf(self,gpt_model):
            uploaded_file = st.file_uploader("File upload", type="pdf")
            if uploaded_file is not None:
                # Create a temporary file to save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

            # Process the uploaded document
            if uploaded_file is not None:
                # Read the PDF
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                print(len(docs))

                st.write("Document successfully uploaded and processed.")

                # Input for the question
                question = st.text_input("Enter your question:")

                llm = ChatOpenAI(model=gpt_model)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                retriever = vectorstore.as_retriever()

                if question:
                    # Load QA Chain with OpenAI (ChatGPT or GPT-4)
                # Load the QA chain
                    qa_chain = load_qa_chain(llm, chain_type="stuff")  # "stuff" is for concatenating text

                    # # Example question
                    # question = "What happen on June 10, 1891 "

                    # Run the question-answering chain
                    response = qa_chain.run(input_documents=splits, question=question)

                    # Output the answer
                    print("Answer:", response) 
                    # Display the answer
                    st.write("Answer:", response)