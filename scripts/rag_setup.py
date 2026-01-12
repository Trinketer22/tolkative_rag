from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.json import load_json_dump
from config import settings
import sys


def help():
    print(f"{sys.argv[0]} <path to docs.jsonl>")
    sys.exit(-1)


def main():
    if len(sys.argv) < 2:
        print("Not enough arguments")
        help()

    uniq_docs = {}
    print("Loading documents...")
    docs = list(
        map(
            lambda doc: Document(
                id=doc["id"], page_content=doc["page_content"], metadata=doc["metadata"]
            ),
            load_json_dump(sys.argv[1]),
        )
    )

    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, cache_folder=settings.MODEL_CACHE_DIR
    )

    # Last doc wins in id clash
    for doc in docs:
        uniq_docs[doc.id] = doc

    clean_docs = uniq_docs.values()

    print("Building main index...")
    full_storage = FAISS.from_documents(clean_docs, embedding=embedder)
    full_storage.save_local(settings.INDEX_PATH)

    print("Building crumbs index...")
    doc_crumbs = map(
        lambda doc: Document(page_content=doc.metadata["crumbs"], id=doc.id), clean_docs
    )
    top_lvl_idx = FAISS.from_documents(list(doc_crumbs), embedding=embedder)
    top_lvl_idx.save_local(settings.HEADERS_INDEX_PATH)

if __name__ == "__main__":
    main()
