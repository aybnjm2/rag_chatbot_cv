import os
import re
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)
embedding_model = GoogleGenerativeAIEmbeddings ( model="models/gemini-embedding-001", google_api_key=os.getenv('GOOGLE_API_KEY'))

# init llm gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0
)

def load_cv(path):
    loader = PyPDFLoader(path)
    pages = loader.load() 
    texte_brut = "\n".join([page.page_content for page in pages])
    return texte_brut


def clean_text(text):
    """Nettoyer le texte : espaces, sauts de ligne, caractères parasites."""
    if not isinstance(text, str):
        text = str(text)
        
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def chunk_document(text, chunk_size,overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( #init le text splitter
        encoding_name='o200k_base',
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.create_documents([text]) # text => chunks
    return chunks


def build_embeddings(chunks, model):  # j'ai commente cette fonction et retourner juste le model puisque chromaDB peut stocker et creer les embeddings en meme temps (mentionne dans la ligne 56)
    # embedding_model = GoogleGenerativeAIEmbeddings ( model=model, google_api_key=os.getenv('GOOGLE_API_KEY'))  ==>  initialisation du modele d'embeddings
    # output_vector = embedding_model.embed_query(chunks) 
    return model


def create_vector_store(chunks, embeddings):     # stocker les vecteurs dans une base vectorielle ChromaDB
    persist_dir = "./chroma_cv_store"

    vectorstore = Chroma.from_documents(     # chromaDB calcule automatiquement l'embedding de chaque chunk
        documents=chunks,
        embedding=embeddings,
        collection_name="cv_active",
        persist_directory=persist_dir
    )
    return vectorstore

def retrieve_relevant_chunks(question,store, k):
    retriever = store.as_retriever(     # creer le retriever de ChromaDB qui va faire la recherche de similarite (cos similarity)
        search_type='similarity',
        search_kwargs={'k': k}
    )
    
    docs_recuperes = retriever.invoke(question)
    return docs_recuperes


def build_prompt(question, context):
    prompt_template = """
    Tu es l'assistant virtuel personnel de l'étudiant à qui appartient ce CV.
    Réponds à la question du recruteur EN TE BASANT UNIQUEMENT sur le contexte fourni issu de son CV.
    
    Règles de sécurité absolues :
    - Si la réponse ne figure pas dans le CV (le contexte ci-dessous), tu DOIS répondre : "Je ne peux pas répondre à cette question car l'information ne se trouve pas dans mon CV."
    - Ne réponds JAMAIS à partir de tes connaissances générales.
    - Sois professionnel, courtois et précis.
    - Ne donne que des informations véridiques issues du texte.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>
    """

    return  prompt_template.format(context=context, question=question)


def generate_answer(prompt):
    response = llm.invoke(prompt)
    return response.content


def answer_cv_only(question, store):
    chunks = retrieve_relevant_chunks(question, store, k=4)
    context = "\n\n".join([doc.page_content for doc in chunks])
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)
    return answer


def reload_vector_store(file): 
    """Gere le rechargement total lorsqu'un nouveau CV est uploade."""
    file_path = "./" + file.name                                
    with open(file_path, "wb") as f:                            
        f.write(file.getbuffer())
        
    try:
        raw_text = load_cv(file_path)                               
        cleaned_text = clean_text(raw_text)
        chunks = chunk_document(cleaned_text, chunk_size=300, overlap=20)
        embeddings = build_embeddings(chunks, model=embedding_model)
        store = create_vector_store(chunks, embeddings)
        return store
    finally:
        if os.path.exists(file_path):
            os.remove(file_path) 


def run_streamlit_app():
    """Dialogue utilisateur via Streamlit (Interface)."""
    st.set_page_config(page_title="Chatbot CV (RAG)", page_icon="📄")
    
    st.title("Chatbot CV Interactif (RAG)")
    st.markdown("Posez vos questions, l'IA vous répondra **uniquement** avec les informations de mon CV.")

    if "store" not in st.session_state:
        st.session_state.store = None
    if "current_cv" not in st.session_state:
        st.session_state.current_cv = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Gestion du CV")
        uploaded_file = st.file_uploader("Chargez votre CV", type=['pdf'])

        if uploaded_file is not None:
            # Si c'est un nouveau fichier, on recharge tout
            if st.session_state.current_cv != uploaded_file.name:
                with st.spinner("Analyse du CV et génération de la base vectorielle..."):
                    st.session_state.store = reload_vector_store(uploaded_file)
                    st.session_state.current_cv = uploaded_file.name
                    st.session_state.messages = [] # On vide l'historique de discussion
                st.success("Base vectorielle mise à jour !")
            
            st.info(f"**CV Actuel :** {st.session_state.current_cv}")
        else:
            st.warning("Veuillez charger un fichier pour démarrer.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Quelles sont les compétences en Python ?"):
        if st.session_state.store is None:
            st.error("Veuillez d'abord charger un CV dans le menu de gauche.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Recherche dans le CV..."):
                    reponse = answer_cv_only(prompt, st.session_state.store)
                    st.markdown(reponse)
            
            st.session_state.messages.append({"role": "assistant", "content": reponse})

if __name__ == "__main__":
    run_streamlit_app()
