import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import tempfile

class SQLRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize lightweight LLM
        self.llm = pipeline('text-generation', model='gpt2', max_length=512)
        # Vector store
        self.documents = []
        self.embeddings = []
        # SQL test case template
        self.test_case_template = """
        Test Case: {test_name}
        Description: {description}
        SQL Query: {sql_query}
        Expected Result: {expected_result}
        """

    def process_documents(self, documents: List[str]) -> None:
        """Process SQL-related documents and store their embeddings."""
        self.documents = documents
        if self.documents:
            self.embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        else:
            self.embeddings = []

    def retrieve_relevant(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve k most relevant documents for a query."""
        if not self.documents or not self.embeddings.any():
            return []
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
            # Compute cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            # Get top k indices, ensuring k doesn't exceed available documents
            k = min(k, len(self.documents))
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return [(self.documents[i], similarities[i]) for i in top_k_indices]
        except Exception as e:
            st.error(f"Error in retrieval: {str(e)}")
            return []

    def generate_sql_query(self, query: str, context: List[str]) -> str:
        """Generate SQL query based on user query and retrieved context."""
        try:
            prompt = f"""
            Given the following context about SQL databases:
            {''.join(context)}
            
            Generate a SQL query for: {query}
            Format the query in a code block:
            ```sql
            query_here
            ```
            """
            response = self.llm(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
            sql_match = re.search(r'```sql\n(.*?)```', response, re.DOTALL)
            return sql_match.group(1).strip() if sql_match else response.strip()
        except Exception as e:
            return f"Error generating SQL query: {str(e)}"

    def generate_test_case(self, sql_query: str, description: str) -> str:
        """Generate a test case for a given SQL query."""
        try:
            expected_result = self._infer_expected_result(sql_query)
            return self.test_case_template.format(
                test_name=f"Test_{hash(sql_query) % 10000}",
                description=description,
                sql_query=sql_query,
                expected_result=expected_result
            )
        except Exception as e:
            return f"Error generating test case: {str(e)}"

    def _infer_expected_result(self, sql_query: str) -> str:
        """Infer expected result for a SQL query."""
        if 'SELECT' in sql_query.upper():
            return "Returns rows matching the query conditions"
        elif 'INSERT' in sql_query.upper():
            return "Successfully inserts record(s) into the table"
        elif 'UPDATE' in sql_query.upper():
            return "Updates the specified records"
        elif 'DELETE' in sql_query.upper():
            return "Deletes the specified records"
        return "Query executes successfully"

    def analyze_sql(self, query: str, documents: List[str] = None) -> Dict:
        """Analyze SQL query and generate test case."""
        if documents:
            self.process_documents(documents)
        relevant_docs = self.retrieve_relevant(query)
        context = [doc for doc, _ in relevant_docs]
        sql_query = self.generate_sql_query(query, context)
        test_case = self.generate_test_case(sql_query, query)
        return {
            "sql_query": sql_query,
            "test_case": test_case,
            "relevant_context": context
        }

# Streamlit app
st.title("SQL RAG Query System")

# File uploader for text documents
uploaded_file = st.file_uploader("Upload a text file with SQL schema/docs", type="txt")

# Initialize session state
if 'sql_rag' not in st.session_state:
    st.session_state.sql_rag = SQLRAG()
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Process uploaded file
if uploaded_file is not None and not st.session_state.document_processed:
    with st.spinner("Processing document..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
                tmp_file.close()  # Explicitly close the file to avoid PermissionError
            
            # Read and process document
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents = [doc.strip() for doc in content.split('\n') if doc.strip()]
            
            if not documents:
                st.session_state.error_message = "Error: No valid content extracted from the file."
                st.error(st.session_state.error_message)
            else:
                # Process documents in SQLRAG
                st.session_state.sql_rag.process_documents(documents)
                st.session_state.document_processed = True
                st.session_state.error_message = None
                st.success("Document processed successfully!")
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except PermissionError:
                st.warning("Could not delete temporary file. It will be cleaned up on system restart.")
            except Exception as e:
                st.warning(f"Error deleting temporary file: {str(e)}")
        except Exception as e:
            st.session_state.error_message = f"Error processing document: {str(e)}"
            st.error(st.session_state.error_message)

# Query interface
if st.session_state.document_processed:
    query = st.text_input("Enter your SQL-related question (e.g., 'Find all users with orders over $100'):")
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Generating SQL query and test case..."):
                try:
                    result = st.session_state.sql_rag.analyze_sql(query)
                    
                    st.subheader("Generated SQL Query:")
                    st.code(result["sql_query"], language="sql")
                    
                    st.subheader("Generated Test Case:")
                    st.text(result["test_case"])
                    
                    st.subheader("Relevant Context:")
                    if result["relevant_context"]:
                        for i, doc in enumerate(result["relevant_context"], 1):
                            with st.expander(f"Context {i}"):
                                st.write(doc)
                    else:
                        st.write("No relevant context found.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query.")
else:
    if uploaded_file is not None and not st.session_state.document_processed:
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        else:
            st.info("Please wait for the document to finish processing.")
    else:
        st.info("Please upload a text file with SQL schema/docs to begin.")