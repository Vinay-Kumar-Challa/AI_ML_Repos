import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

class AIAgent:
    def __init__(self):
        # Initialize embedding model for retrieval
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize lightweight LLM for generation
        self.llm = pipeline('text-generation', model='gpt2', max_length=512)
        # Knowledge base
        self.documents = []
        self.embeddings = []

    def load_documents(self, file_path: str) -> bool:
        """Load and process documents from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into sentences/paragraphs
                self.documents = [doc.strip() for doc in content.split('\n') if doc.strip()]
            if not self.documents:
                return False
            # Generate embeddings
            self.embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
            return True
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return False

    def retrieve_relevant(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve k most relevant documents for a query."""
        if not self.documents:
            return []
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
            # Compute cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            # Get top k indices
            k = min(k, len(self.documents))  # Ensure k doesn't exceed document count
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return [(self.documents[i], similarities[i]) for i in top_k_indices]
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response based on query and context."""
        try:
            prompt = f"""
            Context:
            {''.join(context)}
            
            Question: {query}
            Answer:
            """
            response = self.llm(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
            # Extract answer after "Answer:"
            answer_start = response.find("Answer:")
            if answer_start != -1:
                return response[answer_start + len("Answer:"):].strip()
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_query(self, query: str) -> dict:
        """Process a query and return answer with context."""
        relevant_docs = self.retrieve_relevant(query)
        context = [doc for doc, _ in relevant_docs]
        answer = self.generate_response(query, context)
        return {
            "answer": answer,
            "context": context,
            "query": query
        }

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = AIAgent()
    
    # Load sample document
    sample_file = "sample_knowledge.txt"
    sample_content = """
    The company has a database with a table 'employees' containing columns: id (INT), name (VARCHAR), department (VARCHAR), salary (DECIMAL).
    The 'departments' table includes: dept_id (INT), dept_name (VARCHAR).
    The average salary in the IT department is $75,000.
    """
    
    # Create sample file
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    # Load documents
    if agent.load_documents(sample_file):
        print("Documents loaded successfully!")
        
        # Example queries
        queries = [
            "What is the average salary in the IT department?",
            "What columns are in the employees table?"
        ]
        
        for query in queries:
            result = agent.answer_query(query)
            print(f"\nQuery: {result['query']}")
            print(f"Answer: {result['answer']}")
            print("Relevant Context:")
            for doc in result['context']:
                print(f"- {doc}")
        
        # Clean up sample file
        try:
            os.unlink(sample_file)
        except Exception as e:
            print(f"Error deleting sample file: {str(e)}")
    else:
        print("Failed to load documents.")