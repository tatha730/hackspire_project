
from langchain_core.documents import Document


import os
from langchain_community.document_loaders import TextLoader


## loading pdf files to train the model

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    "data/pdf",
    glob = "**/*.pdf", 
    loader_cls = PyMuPDFLoader,
    show_progress = False
)

pdf_documents = dir_loader.load()
pdf_documents

###chunking 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=2500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better RAG performance.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs

# ...existing code...
import gc
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def is_chunked(docs):
    return bool(docs and isinstance(docs[0].metadata, dict) and docs[0].metadata.get("chunk_id"))

# Inspect current variables
print("Has variable 'chunks' in globals():", 'chunks' in globals())
print("Is 'chunks' chunked?:", is_chunked(globals().get('chunks')) if 'chunks' in globals() else False)
print("Is 'pdf_documents' chunked?:", is_chunked(globals().get('pdf_documents')) if 'pdf_documents' in globals() else False)

# Delete in-memory chunked objects
if 'chunks' in globals():
    del chunks
    print("Deleted variable 'chunks'")

# If pdf_documents was overwritten with chunks, reload originals from PDF folder
pdf_dir = "../data/pdf"
if 'pdf_documents' in globals() and is_chunked(pdf_documents):
    print("pdf_documents looks chunked — reloading original PDFs from", pdf_dir)
    dir_loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )
    pdf_documents = dir_loader.load()
    print("Reloaded pdf_documents, count =", len(pdf_documents))

# force python to free memory
gc.collect()
print("Cleanup done.")
# ...existing code...

chunks=chunk_documents(pdf_documents)
chunks

### Embedding and vectorstore

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    def __init__(self,model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: Huggingface model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

## initialize the embedding manager

embedding_manager = EmbeddingManager()
embedding_manager

### VectorStore


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        """Initialize the vector store
        
        Args:
            collection_name Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # create persistent Chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match nuber of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents tpo vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore=VectorStore()
vectorstore
        

texts=[doc.page_content for doc in chunks]

embeddings = embedding_manager.generate_embeddings(texts)

vectorstore.add_documents(chunks, embeddings)

### Retriever pipeline from VectorStore
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'if': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("no documents found")
            
            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
rag_retriever=RAGRetriever(vectorstore,embedding_manager)
        


rag_retriever

rag_retriever.retrieve("i feel depressed ")

rag_retriever.retrieve("what is the distance between earth and moon")

from datetime import datetime
import re
from typing import List

def preprocess_query(query: str) -> List[str]:
    """Create variations of the query to improve matching."""
    variations = [query]
    
    # Convert to lowercase
    query_lower = query.lower()
    if query_lower != query:
        variations.append(query_lower)
    
    # Remove punctuation
    clean_query = re.sub(r'[.,?!]', '', query_lower)
    if clean_query != query_lower:
        variations.append(clean_query)
    
    # Remove stop words for core concept matching
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by'}
    words = query_lower.split()
    core_query = ' '.join([w for w in words if w not in stop_words])
    if core_query != query_lower:
        variations.append(core_query)
    
    return list(set(variations))  # Remove duplicates

def save_user_query(query: str, metadata: dict = None, response: str = None):
    """Embed and save a user query with variations to improve retrieval probability."""
    if metadata is None:
        metadata = {}
    
    # Get query variations
    query_variations = preprocess_query(query)
    
    # Enhanced metadata
    base_metadata = {
        'type': 'user_query',
        'original_query': query,
        'timestamp': datetime.utcnow().isoformat(),
        'query_length': len(query),
        'word_count': len(query.split()),
        'is_question': '?' in query,
    }
    
    if response:
        base_metadata['response'] = response
        base_metadata['interaction_complete'] = True
    
    # Merge with user provided metadata
    base_metadata.update(metadata)
    
    # Create documents for each variation
    documents = []
    embeddings = []
    
    # Generate embeddings for all variations at once (more efficient)
    all_embeddings = embedding_manager.generate_embeddings(query_variations)
    
    for var, emb in zip(query_variations, all_embeddings):
        # Create a document with the variation
        var_metadata = base_metadata.copy()
        var_metadata['is_variation'] = (var != query)
        var_metadata['variation_type'] = 'original' if var == query else 'processed'
        
        doc = Document(page_content=var, metadata=var_metadata)
        documents.append(doc)
        embeddings.append(emb)
    
    # Add all variations to vector store
    vectorstore.add_documents(documents, np.array(embeddings))
    print(f"Saved user query: '{query}' with {len(query_variations)} variations")

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = "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"

llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant",temperature=0.1,max_tokens=1024)
def rag_simple(query,retriever,llm,top_k=3):
    
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        save_user_query(query)
        return "No relevant context found to answer the question."
    
    prompt=f""""You are a concise, empathetic mental health support assistant. Respond to the user's "
        "message using the provided context when available. Keep responses clear, direct, "
        "and focused on supporting the user's emotional well-being.
        Context:
        {context}

        Question: {query}

        Answer:"""
    
    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content


import random
import re
from datetime import datetime
from typing import List, Tuple, Dict

history = []
MAX_TURNS = 6  # Keep last 3 user+assistant pairs

SYSTEM_INSTRUCTION = (
    "You are a concise, empathetic mental health support assistant. Respond to the user's "
    "message using the provided context when available. Keep responses clear, direct, "
    "and focused on supporting the user's emotional well-being."
)

def sanitize_output(text: str) -> str:
    """Clean the model output for consistent formatting."""
    if not text:
        return text

    # Remove common artifacts and meta markers
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove role markers like 'Assistant:' if present
    text = re.sub(r"\bAssistant:\b", "", text)

    return text.strip()

def enhance_prompt_with_history(query: str, context: str, chat_history: List[Tuple[str, str]]) -> str:
    """Create a prompt that includes conversation history and context."""
    # Include last few turns of conversation for context
    history_text = ""
    if chat_history:
        last_turns = chat_history[-3:]  # Get last 3 turns
        for user_msg, assistant_msg in last_turns:
            history_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"

    # Construct the prompt
    prompt = f"""{SYSTEM_INSTRUCTION}

Previous Conversation:
{history_text}
Relevant Context:
{context}

User: {query}
Assistant:"""

    return prompt

def get_fallback_response() -> str:
    """Get a fallback response when the model output is insufficient."""
    fallbacks = [
        "I understand this is difficult. Could you tell me more about what you're experiencing?",
        "I'm here to listen and support you. What's been on your mind?",
        "Thank you for sharing. Can you help me understand what led to these feelings?",
        "I'm here with you. Would you like to elaborate on what you're going through?"
    ]
    return random.choice(fallbacks)

def rag_chat(query: str, chat_history: List[Tuple[str, str]], retriever=None, llm=None) -> str:
    """Enhanced RAG with conversation history.

    NOTE: This function returns the assistant response ONLY and does NOT mutate
    the provided chat_history. The caller (interactive loop) is responsible for
    appending the (user, assistant) pair to chat_history and for printing the
    assistant output. This prevents duplicate prints or duplicate history entries.
    """
    # Default to the notebook-level rag_retriever/llm if not passed in
    if retriever is None:
        try:
            retriever = rag_retriever
        except NameError:
            retriever = None
    if llm is None:
        try:
            llm = llm
        except NameError:
            llm = None

    # Try to find relevant context from previous interactions if retriever is available
    context = ""
    if retriever is not None:
        try:
            results = retriever.retrieve(query, top_k=3, score_threshold=0.7)
            context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        except Exception:
            context = ""

    # If no relevant context found and retriever existed, save the query for future reference
    if not context and retriever is not None:
        try:
            save_user_query(query)
        except Exception:
            pass

    # Create prompt with history and context
    prompt = enhance_prompt_with_history(query, context, chat_history)

    # Generate response via llm
    if llm is None:
        # No LLM available; return a fallback
        return get_fallback_response()

    try:
        response = llm.invoke([prompt])
        # response may be an object; prefer `.content` if present
        response_text = getattr(response, 'content', response)
        if isinstance(response_text, (list, tuple)):
            response_text = str(response_text[0])
        response_text = sanitize_output(response_text)

        # Use fallback if response is too short or empty
        if len(response_text.split()) < 3:
            response_text = get_fallback_response()

        # Avoid exact repetition with last response in chat_history
        if chat_history and chat_history[-1][1] == response_text:
            response_text = get_fallback_response()

        return response_text

    except Exception:
        return get_fallback_response()

def interactive_chat():
    """Run an interactive chat session.

    This loop is the single place that appends to and prints chat_history. By
    centralizing history mutation here we avoid duplicate entries or duplicate
    prints that can occur if rag_chat also mutates history or prints output.
    """
    chat_history: List[Tuple[str, str]] = []
    print("Mental Health Support Bot")
    print("Type 'quit' to exit, 'reset' to start over\n")

    try:
        while True:
            message = input("You: ").strip()

            if message.lower() == "quit":
                print("\nBot: Take care! 💙")
                break
            elif message.lower() == "reset":
                chat_history = []
                print("Conversation reset.\n")
                continue
            elif not message:
                continue

            # Get a single assistant response (rag_chat does not modify chat_history)
            assistant_response = rag_chat(message, chat_history, retriever=rag_retriever, llm=llm)

            # Append the user/assistant pair exactly once here
            chat_history.append((message, assistant_response))

            # Trim history if needed
            if len(chat_history) > MAX_TURNS:
                chat_history = chat_history[-MAX_TURNS:]

            # Print the assistant response exactly once
            print(f"\nBot: {assistant_response}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nBot: Take care! 💙")

# Example usage
#if __name__ == "__main__":
#    interactive_chat()