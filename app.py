import streamlit as st

def main():
    st.title("Document-Aware RAG Chatbot")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    st.caption(source)

    # Sidebar for controls
import streamlit as st
from utils.document_parser import parse_pdf, parse_docx, parse_txt
from core.rag_pipeline import chunk_texts, create_vector_store, get_answer_from_rag, embeddings_model # Import embeddings_model
from integrations.web_retriever import fetch_url_content # Import web retriever
from integrations.wikipedia_connector import fetch_wikipedia_content # Import Wikipedia connector
import os

# ... (keep existing imports like import streamlit as st)

def main():
    st.title("Document-Aware RAG Chatbot")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    st.caption(source)

    # Sidebar for controls
    st.sidebar.title("Controls")

    # Export Chat Button
    if 'messages' in st.session_state and st.session_state.messages:
        chat_export_data = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" +
            ("\nSources:\n" + "\n".join([f"  - {s}" for s in msg.get('sources', [])]) if msg.get('sources') else "")
            for msg in st.session_state.messages
        ])
        st.sidebar.download_button(
            label="Export Chat as TXT",
            data=chat_export_data,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    
    st.sidebar.markdown("---") # Separator
    st.sidebar.subheader("Wikipedia Search")
    use_wikipedia = st.sidebar.checkbox("Enable Wikipedia Search", value=True, key="use_wikipedia")
    num_wiki_results = st.sidebar.slider("Number of Wikipedia Results", 1, 5, 1, disabled=not use_wikipedia)

    st.sidebar.markdown("---") # Separator
    st.sidebar.subheader("Add Content from Web URL")
    url_input = st.sidebar.text_input("Enter URL to fetch and index:", key="url_input")
    fetch_url_button = st.sidebar.button("Fetch & Index URL")

    if fetch_url_button and url_input:
        with st.spinner(f"Fetching and processing {url_input}..."):
            retrieved_content = fetch_url_content(url_input)
            if retrieved_content and retrieved_content.get("text"):
                st.sidebar.success(f"Successfully fetched content from {url_input[:50]}...")
                web_doc = [{'name': url_input, 'content': retrieved_content["text"]}]
                web_text_chunks_with_metadata = chunk_texts(web_doc)

                if web_text_chunks_with_metadata:
                    if 'vector_store' in st.session_state and st.session_state['vector_store']:
                        try:
                            texts_to_add = [chunk['page_content'] for chunk in web_text_chunks_with_metadata]
                            metadatas_to_add = [chunk['metadata'] for chunk in web_text_chunks_with_metadata]
                            st.session_state['vector_store'].add_texts(texts=texts_to_add, metadatas=metadatas_to_add, embedding=embeddings_model)
                            st.sidebar.success(f"Content from {url_input[:30]}... added to knowledge base.")
                        except Exception as e:
                            st.sidebar.error(f"Error adding URL content to vector store: {e}")
                    else:
                        # Create a new vector store if one doesn't exist
                        vector_store = create_vector_store(web_text_chunks_with_metadata)
                        if vector_store:
                            st.session_state['vector_store'] = vector_store
                            st.sidebar.success(f"Knowledge base created with content from {url_input[:30]}...")
                        else:
                            st.sidebar.error("Failed to create knowledge base from URL. Check Hugging Face Hub API token (if required) and logs.")
                            if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                                st.sidebar.warning("HUGGINGFACEHUB_API_TOKEN not set. Some models may require it.")
                else:
                    st.sidebar.warning(f"No text could be extracted from {url_input} to add to knowledge base.")
            else:
                st.sidebar.error(f"Failed to fetch content from {url_input}. Please check the URL and try again.")

    st.sidebar.markdown("---") # Separator
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents (.pdf, .docx, .txt)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")
        parsed_documents = []
        for uploaded_file in uploaded_files:
            st.sidebar.write(f"Processing: {uploaded_file.name}")
            file_content = uploaded_file.getvalue()
            file_extension = uploaded_file.name.split('.')[-1].lower()
            text = ""
            if file_extension == "pdf":
                text = parse_pdf(file_content)
            elif file_extension == "docx":
                text = parse_docx(file_content)
            elif file_extension == "txt":
                text = parse_txt(file_content)
            
            if text:
                parsed_documents.append({"name": uploaded_file.name, "content": text})
                st.sidebar.write(f"Successfully parsed: {uploaded_file.name} ({len(text)} chars)")
            else:
                st.sidebar.error(f"Could not parse: {uploaded_file.name}")
        
        if parsed_documents:
            st.session_state['parsed_docs'] = parsed_documents
            # Create vector store once documents are parsed
            with st.spinner("Processing documents and building knowledge base..."):
                all_text_chunks_with_metadata = chunk_texts(parsed_documents)
                if all_text_chunks_with_metadata:
                    vector_store = create_vector_store(all_text_chunks_with_metadata)
                    if vector_store:
                        st.session_state['vector_store'] = vector_store
                        st.sidebar.success("Knowledge base created successfully!")
                    else:
                        st.sidebar.error("Failed to create knowledge base. Check Hugging Face Hub API token (if required) and logs.")
                        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                            st.sidebar.warning("HUGGINGFACEHUB_API_TOKEN not set. Some models may require it. Please set it as an environment variable if needed.")
                else:
                    st.sidebar.warning("No text could be extracted from the documents to build a knowledge base.")

    # Main chat interface area
    st.header("Chat with your documents and Wikipedia")

    # Chat input using st.chat_input
    if user_query := st.chat_input("Ask a question about your documents:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process the query and get the RAG response
        # (The RAG logic block below will be executed here)
        # st.write(f"**You asked:** {user_query}") # This is now handled by chat_message("user")
        if 'vector_store' in st.session_state and st.session_state['vector_store']:
            with st.spinner("Searching for answers..."):
                wikipedia_data = None
                if st.session_state.get('use_wikipedia', True):
                    st.sidebar.write("Fetching from Wikipedia...")
                    wikipedia_data = fetch_wikipedia_content(user_query, num_results=num_wiki_results)
                    if wikipedia_data:
                        st.sidebar.success(f"Fetched {len(wikipedia_data)} article(s) from Wikipedia.")
                    else:
                        st.sidebar.info("No relevant articles found on Wikipedia or search disabled.")

                answer, sources = get_answer_from_rag(user_query, st.session_state['vector_store'], wikipedia_content=wikipedia_data)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown("**Answer:**")
                    st.info(answer)
                    if sources:
                        st.markdown("**Sources:**")
                        for source_item in sources:
                            st.caption(source_item)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        elif not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            # This check is for models that require a token. Some Hugging Face models run locally or don't need a token.
            # We'll provide a general warning that can be adjusted based on the specific model chosen in rag_pipeline.py
            warning_message = "HUGGINGFACEHUB_API_TOKEN not found. Some Hugging Face models may require this token. Please set it as an environment variable if you encounter issues."
            # For a less intrusive warning, consider logging this or making it a less prominent UI element.
            # For now, we'll keep a similar error display for consistency.
            with st.chat_message("assistant"):
                st.warning(warning_message) # Changed to warning as not all HF models need a token
            st.session_state.messages.append({"role": "assistant", "content": warning_message})
        else:
            warning_message = "Please upload and process documents first to build the knowledge base."
            with st.chat_message("assistant"):
                st.warning(warning_message)
            st.session_state.messages.append({"role": "assistant", "content": warning_message})
        
        # Optionally, keep the display of parsed document content for debugging if needed
        # if 'parsed_docs' in st.session_state and False: # Disabled for cleaner UI
        #     st.subheader("Parsed Document Content (First 500 chars each):")
        #     for doc in st.session_state['parsed_docs']:
        #         st.text_area(doc['name'], doc['content'][:500], height=100, disabled=True)

if __name__ == "__main__":
    main()