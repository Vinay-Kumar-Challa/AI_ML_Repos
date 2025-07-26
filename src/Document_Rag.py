import streamlit as st
#import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
#from transformers import pipeline



# Streamlit app
st.title("RAG Document Query System")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")


# Function to create RAG pipeline
def create_rag_pipeline(document_path):
    try:
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        texts = text_splitter.split_documents(documents)
        print(len(texts))
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
            )
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        print(vector_store)
        
        # Initialize LLM
        llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs = {"max_length":512, "max_new_tokens":512}
            )
        
        print(llm)
        
        k = min(1, len(texts))
        
        # Create RAG pipeline
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
            )
        
        return qa_chain, None
    except Exception as e:
        return None , str(e)
    
    
# Function to query document
def query_document(qa_chain, query):
    try:
        result = qa_chain.invoke({"query": query})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }, None
    except Exception as e:
        return None, str(e)
        

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain=None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = None
    
# Process uploaded file
if uploaded_file is not None and not st.session_state.document_processed:
    with st.spinner("Processing Document"):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
            
            # Create RAG pipeline
            qa_chain, error = create_rag_pipeline(tmp_file_path)
            
            # Clean up temporary file
            #os.unlink(tmp_file_path)
            
            if error:
                st.error(f"Error processing document: {error}")
            else:
                st.session_state.qa_chain = qa_chain
                st.session_state.document_processed = True
                st.success("Document processed successfully!")



    
    
# Query interface
if st.session_state.document_processed and st.session_state.qa_chain is not None:
    query = st.text_input("Enter your question about the document:")
    if st.button("Submit Query"):
        if query:
            with st.spinner("Generating answer..."):
                response, error = query_document(st.session_state.qa_chain, query)
                
                if error:
                    st.error(f"Error processing query: {error}")
                else:
                    st.subheader("Answer:")
                    st.write(response["answer"])
                    
                    st.subheader("Source Documents:")
                    for i, doc in enumerate(response["source_documents"], 1):
                        with st.expander(f"Source {i}"):
                            st.write(doc.page_content)
        else:
            st.warning("Please enter a query.")
else:
    if uploaded_file is not None and not st.session_state.document_processed:
        st.info("Please wait for the document to finish processing.")
    else:
        st.info("Please upload a PDF document to begin.")