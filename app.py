import hashlib
import logging

import streamlit as st

from rag import build_store_from_pdf, answer_cv_only, get_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 10


def init_state():
    defaults = {
        "store": None,
        "current_cv": None,
        "current_cv_hash": None,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_upload(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if file_hash == st.session_state.current_cv_hash:
        return  # même contenu, rien à refaire

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Fichier trop volumineux (max {MAX_UPLOAD_MB} Mo).")
        return
    if not file_bytes.startswith(b"%PDF-"):
        st.error("Ce fichier ne semble pas être un PDF valide.")
        return

    with st.spinner("Analyse du CV et génération de la base vectorielle..."):
        try:
            new_store = build_store_from_pdf(
                file_bytes, collection_name=f"cv_{file_hash[:16]}"
            )
        except Exception:
            logger.exception("Échec du traitement du CV")
            st.error("Impossible de traiter ce CV. Vérifiez le PDF et réessayez.")
            return

    # Supprimer l'ancienne collection seulement après succès
    old_store = st.session_state.store
    if old_store is not None:
        try:
            old_store.delete_collection()
        except Exception:
            logger.warning("Suppression de l'ancienne collection impossible")

    st.session_state.store = new_store
    st.session_state.current_cv_hash = file_hash
    st.session_state.current_cv = uploaded_file.name
    st.session_state.messages = []
    st.success("Base vectorielle mise à jour !")


def run_streamlit_app():
    st.set_page_config(page_title="Chatbot CV (RAG)", page_icon="")

    try:
        get_api_key()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    init_state()

    st.title("Chatbot CV Interactif (RAG)")
    st.markdown(
        "Posez vos questions, l'IA vous répondra **uniquement** "
        "avec les informations du CV."
    )

    with st.sidebar:
        st.header("Gestion du CV")
        uploaded_file = st.file_uploader("Chargez votre CV", type=["pdf"])

        if uploaded_file is not None:
            process_upload(uploaded_file)
            if st.session_state.current_cv:
                st.info(f"**CV actuel :** {st.session_state.current_cv}")
        else:
            st.warning("Veuillez charger un fichier pour démarrer.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Quelles sont les compétences en Python ?"):
        prompt = prompt.strip()
        if not prompt:
            st.warning("Veuillez saisir une question.")
        elif st.session_state.store is None:
            st.error("Veuillez d'abord charger un CV dans le menu de gauche.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Recherche dans le CV..."):
                    try:
                        reponse, sources = answer_cv_only(
                            prompt, st.session_state.store
                        )
                    except Exception:
                        logger.exception("Erreur lors de la génération")
                        st.error("Une erreur est survenue. Réessayez plus tard.")
                        st.stop()

                st.markdown(reponse)
                if sources:
                    with st.expander("Sources"):
                        for doc in sources:
                            page = doc.metadata.get("page")
                            label = f"Page {page + 1}" if isinstance(page, int) else "Page ?"
                            st.caption(label)
                            st.text(doc.page_content[:300])

            # Historique mis à jour seulement si tout s'est bien passé
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reponse})


if __name__ == "__main__":
    run_streamlit_app()