import os
import logging
from backend import load_and_chunk_codebase, ingest_into_chroma
from config import WORKSPACE_ROOT, CHROMA_DB_DIR

# Configure logging to see results in terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("🚀 --- Starting Full Codebase Ingestion ---")
    print(f"Target Directory: {WORKSPACE_ROOT}")
    print(f"Database Directory: {CHROMA_DB_DIR}")
    print("-" * 45)

    if not os.path.exists(WORKSPACE_ROOT):
        logger.error(f"Error: Workspace root '{WORKSPACE_ROOT}' does not exist.")
        return

    try:
        # 1. Load and Chunk
        logger.info("📡 Scanning and chunking codebase...")
        def progress_callback(curr, total, filename):
            if curr % 5 == 0 or curr == total:
                print(f"Progress: [{curr}/{total}] - Processing: {filename}")

        chunks = load_and_chunk_codebase(str(WORKSPACE_ROOT), on_progress=progress_callback)
        
        if not chunks:
            logger.warning("⚠️ No files found matching your CODE_EXTENSIONS in config.py!")
            return

        logger.info(f"✅ Created {len(chunks)} chunks from your code.")

        # 2. Ingest into Chroma
        logger.info("📥 Embedding chunks and saving to ChromaDB (this may take a minute)...")
        db, added_count = ingest_into_chroma(chunks, collection_name="default")

        print("-" * 45)
        print(f"✨ SUCCESS: Added {added_count} new chunks to the database.")
        print(f"📚 Total database size: {db._collection.count()} chunks.")
        print("Done! You can now use the CLI to ask questions about your whole project.")

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
