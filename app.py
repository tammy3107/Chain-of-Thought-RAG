import streamlit as st                                                  
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma                     
from langchain.prompts import ChatPromptTemplate, PromptTemplate        
from langchain_core.output_parsers import StrOutputParser               
from langchain_community.chat_models import ChatOllama                  
from langchain_core.runnables import RunnablePassthrough                

# Setting up a Chroma vector database with embeddings
vector_db = Chroma(
    persist_directory="./chroma_db",                           
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

# Configuring a LLM instance for answering questions
local_model = "llama3.2"    
llm = ChatOllama(model=local_model)          

# Configuring a retriever with a similarity-based search threshold
retriever = vector_db.as_retriever(
    search_type="mmr",                             
)

# Defining a prompt template for RAG
template = """"You are an intelligent assistant with access to relevant information. "
        "Use the provided context to answer the question step-by-step.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Think through the solution in clear, logical steps.\n\n"
        "Step-by-Step Reasoning:"
        "Mention the steps:"
        "Summerise all the steps and give an answer:"
"""

# Setting up the RAG pipeline with the prompt and LLM
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}          
    | prompt                                                            
    | llm                                                               
    | StrOutputParser()                                                
)

# Streamlit UI section
st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

# Initializing session state for message history and vector database
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None

# Assigning the vector database to session state
st.session_state["vector_db"] = vector_db

# Creating a message container with a specified height and border
message_container = st.container(height=500, border=True)
for message in st.session_state["messages"]:
    avatar = "🤖" if message["role"] == "assistant" else "😎"             
    with message_container.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])                                 

# Handling user input prompts through the Streamlit chat interface
if prompt := st.chat_input("Enter a prompt here..."):
    try:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        message_container.chat_message("user", avatar="😎").markdown(prompt)

        # Generating a response with a spinner if the vector database is ready
        with message_container.chat_message("assistant", avatar="🤖"):
            with st.spinner(":green[processing...]"):
                if st.session_state["vector_db"] is not None:
                    response = chain.invoke(prompt)                     
                    st.markdown(response)
                else:
                    st.warning("Knowledge Base is not ready")# Warning if vector database is unavailable

        # Storing the assistant's response in session state
        if st.session_state["vector_db"] is not None:
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
    except Exception as e:
        st.error(e, icon="⛔️")# Handling errors with a message
