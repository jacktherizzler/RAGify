# This file will contain the core RAG pipeline logic.
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# NEW: Import transformers pipeline for local inference
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Using a sentence-transformer model for embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")

# Remove HuggingFaceHub and use local pipeline
# Load Flan-T5 model locally
local_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(local_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Wrapper to use pipeline in LangChain-like interface
class LocalFlanT5:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def __call__(self, prompt, **kwargs):
        result = self.pipeline(prompt, max_length=512, temperature=0.7, **kwargs)
        return result[0]["generated_text"] if result and "generated_text" in result[0] else result[0]["text"]

llm = LocalFlanT5(llm_pipeline)

def chunk_texts(documents: list) -> list:
    """Chunks text from a list of document dictionaries."""
    all_texts = []
    for doc in documents:
        # Assuming doc is a dictionary with 'name' and 'content'
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=doc['content'])
        # Add metadata (source document name) to each chunk
        for i, chunk in enumerate(chunks):
            all_texts.append({'page_content': chunk, 'metadata': {'source': doc['name'], 'chunk_index': i}})
    return all_texts

def create_vector_store(text_chunks_with_metadata: list):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks_with_metadata:
        return None
    
    # Separate texts and metadatas for FAISS
    texts = [chunk['page_content'] for chunk in text_chunks_with_metadata]
    metadatas = [chunk['metadata'] for chunk in text_chunks_with_metadata]
    
    try:
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings_model, metadatas=metadatas)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Potentially handle API key errors or other embedding issues here
        if "api key" in str(e).lower() or "token" in str(e).lower():
            print("Please ensure your HUGGINGFACEHUB_API_TOKEN (if required by the model) is correctly set and valid.")
        return None

def get_answer_from_rag(query: str, vector_store: FAISS, wikipedia_content: list = None):
    """Generates an answer using the RAG pipeline (manual, no LangChain LLMChain)."""
    if not vector_store:
        return "Vector store not available. Please upload and process documents first.", []

    # Retrieve top 3 relevant chunks manually
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    document_context_str = "\n\n".join([doc.page_content for doc in docs])

    # Prepare context from Wikipedia
    wikipedia_context_str = ""
    wikipedia_sources_for_display = []
    if wikipedia_content:
        for item in wikipedia_content:
            wikipedia_context_str += f"\n\n[Wikipedia Content - {item.get('title', 'N/A')}]\n{item.get('summary', '')}"
            wikipedia_sources_for_display.append(f"Wikipedia: {item.get('title', 'N/A')} - {item.get('url', 'N/A')}")

    # Define a prompt template
    prompt_template = """
    You are a helpful AI assistant. Use the following pieces of context from documents and potentially Wikipedia to answer the question at the end. 
    If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
    Prioritize information from uploaded documents if available and relevant. If Wikipedia content is provided, use it to supplement the answer.

    Document Context: {context}
    {wikipedia_context_placeholder}

    Question: {question}
    Helpful Answer:
    """

    # Dynamically add Wikipedia context to the prompt if available
    if wikipedia_context_str:
        final_prompt_template = prompt_template.replace("{wikipedia_context_placeholder}", f"Wikipedia Context:\n{wikipedia_context_str}")
    else:
        final_prompt_template = prompt_template.replace("{wikipedia_context_placeholder}", "")

    # Fill in the prompt
    prompt = final_prompt_template.format(context=document_context_str, question=query)

    try:
        # Call the local Flan-T5 pipeline directly
        answer = llm(prompt)
        # Format document sources for display
        doc_sources_for_display = []
        if docs:
            for doc in docs:
                doc_sources_for_display.append(
                    f"Document: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')}"
                )
        all_sources = doc_sources_for_display + wikipedia_sources_for_display
        return answer, all_sources
    except Exception as e:
        print(f"Error during RAG answer generation: {e}")
        return f"An error occurred: {str(e)}", []