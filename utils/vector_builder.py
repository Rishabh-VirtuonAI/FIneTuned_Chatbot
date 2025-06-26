# utils/vector_builder.py

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

def build_faiss_index(domain: str, kb_file_path: str):
    """
    Build and save FAISS vector index for the specified domain.
    """
    try:
        loader = TextLoader(kb_file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=30,
            length_function=len
        )
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.from_documents(chunks, embedding)
        index_dir = os.path.join("domains", domain, "faiss_index")
        vectorstore.save_local(index_dir)
        logger.info(f"FAISS index built and saved to {index_dir}")

    except Exception as e:
        logger.error(f"Failed to build FAISS index for domain '{domain}': {e}")
        raise
