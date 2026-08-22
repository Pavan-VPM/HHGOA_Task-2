import os
import sys
import json
from pathlib import Path

def bootstrap():
    print("Checking for 'datasets' library...")
    try:
        import datasets
    except ImportError:
        print("Installing 'datasets' library via pip...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pyarrow", "--break-system-packages"])
            import datasets
        except Exception as err:
            print(f"Failed to install dependencies: {err}")
            sys.exit(1)

    print("Loading a stream subset of 'ai4bharat/MSMARCO-XI' (Hindi language subset)...")
    try:
        # Load streaming to avoid downloading gigabytes of data
        ds = datasets.load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train", streaming=True)
    except Exception as exc:
        print(f"Error connecting to Hugging Face: {exc}")
        print("Make sure you have active internet connectivity.")
        sys.exit(1)

    print("Extracting unique passages for indexing...")
    documents = []
    seen_texts = set()
    doc_count = 0
    
    # Process the first 25 query examples and harvest their passages
    iterator = iter(ds)
    for _ in range(25):
        try:
            row = next(iterator)
            query = row.get("query", "")
            passages = row.get("passages", [])
            for p in passages:
                text = p.get("passage_text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    doc_id = f"msmarco-hi-{doc_count}"
                    documents.append({
                        "id": doc_id,
                        "title": f"MSMARCO-XI Hindi Passage {doc_count}",
                        "text": text,
                        "metadata": {
                            "source": "msmarco-xi",
                            "language": "hi",
                            "associated_query": query
                        }
                    })
                    doc_count += 1
        except StopIteration:
            break

    if not documents:
        print("No documents were extracted.")
        sys.exit(1)

    print(f"Extracted {len(documents)} unique passages from MSMARCO-XI.")
    
    # Import RAGVoice modules and run ingestion
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from ragvoice.config import AppConfig
    from ragvoice.ingest import CorpusIngester

    config = AppConfig.from_env()
    print(f"Connecting to database: {config.db_path}")
    
    ingester = CorpusIngester(config)
    result = ingester.ingest_documents(documents)
    print(f"Ingestion successful: {result}")

if __name__ == "__main__":
    bootstrap()
