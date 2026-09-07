import os
import json
from datetime import datetime
from app.memory.semantic_memory import SemanticMemoryService
from app.memory.embeddings import EmbeddingService


def run_phase2():
    print("=== Phase 2: Semantic Memory Pipeline & Vector Storage ===")
    print("=" * 60)

    sample_dir = "tests/fixtures/sample_journals"
    
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory '{sample_dir}' not found.")
        return

    # Load all text files and JSON files
    all_texts = []
    print("\n📂 Loading files from sample directory...")
    
    for filename in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        try:
            if filename.endswith('.txt') or filename.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_texts.append({
                        'text': content,
                        'source': filename,
                        'type': 'text'
                    })
                print(f"   ✅ Loaded '{filename}'")
                
            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'text' in item:
                                all_texts.append({
                                    'text': item['text'],
                                    'source': filename,
                                    'type': 'json'
                                })
                    elif isinstance(data, dict):
                        if 'messages' in data:
                            for msg in data['messages']:
                                if 'content' in msg:
                                    all_texts.append({
                                        'text': msg['content'],
                                        'source': filename,
                                        'type': 'json'
                                    })
                        elif 'text' in data:
                            all_texts.append({
                                'text': data['text'],
                                'source': filename,
                                'type': 'json'
                            })
                print(f"   ✅ Loaded '{filename}'")
                
        except Exception as e:
            print(f"   ❌ Error loading '{filename}': {e}")

    if not all_texts:
        print("❌ No text found in sample files.")
        return

    print(f"\n📊 Total Text Chunks Extracted: {len(all_texts)}")

    # Create chunks from the texts
    chunks = []
    for i, item in enumerate(all_texts):
        text = item['text']
        for j in range(0, len(text), 300):
            chunk_text = text[j:j+300]
            if chunk_text.strip():
                chunks.append({
                    'chunk_id': f"chunk_{i}_{j}",
                    'text': chunk_text,
                    'source_type': item['type'],
                    'source_file': item['source'],
                    'original_date': datetime.now().isoformat()
                })

    print(f"📊 Created {len(chunks)} chunks from sample data.")

    if not chunks:
        print("❌ No chunks created.")
        return

    print("\n🔮 Creating embeddings and storing in ChromaDB...")
    memory_service = SemanticMemoryService()
    
    try:
        indexed_count = memory_service.index_chunks(chunks)
        print(f"✅ Successfully embedded and stored {indexed_count} chunks in ChromaDB.")
    except Exception as e:
        print(f"❌ Error indexing chunks: {e}")
        return

    # Test queries
    test_queries = [
        "What is my preferred coffee order?",
        "What programming languages or technical topics am I working on?",
        "What are my preferences for working hours?",
    ]

    print("\n" + "=" * 60)
    print("🔍 Executing Verification Retrieval Queries")
    print("=" * 60)

    for q in test_queries:
        print(f"\n📝 Query: '{q}'")
        try:
            results = memory_service.retrieve(query=q, k=2)
            if results:
                for i, res in enumerate(results, 1):
                    score = res.get("combined_score", res["similarity"])
                    date = res.get("metadata", {}).get("original_date", "N/A")
                    text_preview = res['text'][:100] + "..." if len(res['text']) > 100 else res['text']
                    print(f"   Result #{i} [Score: {score:.4f}] [Date: {date}]")
                    print(f"   Excerpt: {text_preview}\n")
            else:
                print("   No results found.")
        except Exception as e:
            print(f"   ❌ Query error: {e}")

    print("\n" + "=" * 60)
    print("📊 Phase 2 Complete!")
    print("=" * 60)
    stats = memory_service.vector_store.count()
    print(f"Total chunks in vector store: {stats}")


if __name__ == "__main__":
    run_phase2()
