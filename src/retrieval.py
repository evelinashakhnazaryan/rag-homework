"""Only articles enter the index; golden answers never enter generation."""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index(articles, config):
    documents = [Document(page_content=f"{a.get('title', '')}\n{a['text']}",
                          metadata={"article_id": a["id"], "title": a.get("title", "")})
                 for a in articles]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"],
        add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"], model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "})
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    retriever = store.as_retriever(search_kwargs={"k": config["top_k"]})
    return store, retriever, chunks
