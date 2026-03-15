import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Based on https://youtu.be/YDdKiQNw80c?si=69I0IWrti8llzast&t=600 this script will take a PDF from your data directory (named "example"),
# split it into chunks, create embeddings for those chunks, and then perform a similarity search based on a query specified on the command
# line

def main():
    parser = argparse.ArgumentParser(description="Query a PDF via Vector Search")
    parser.add_argument("query", type=str, help="The question to ask the document")
    parser.add_argument("--file", type=str, default="data/example.pdf", help="Path to PDF")
    args = parser.parse_args()

    print(f"Searching for: {args.query}")

    # Load and split
    loader = PyPDFLoader(args.file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = loader.load_and_split(text_splitter)

    # Embedding and DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)

    # Search
    results = vector_db.similarity_search(args.query, k=2)
    
    if results:
        print(f"\n--- Top Result ---\n{results[0].page_content}\n")

if __name__ == "__main__":
    main()