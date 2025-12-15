#!/usr/bin/env python3
"""
Local Multimodal AI Agent - Main CLI Entry Point
"""
import click
import os
from pathlib import Path
from typing import List, Optional
from app.config import PAPERS_DIR, IMAGES_DIR
from app.embeddings import TextEmbedder, ImageEmbedder
from app.chroma_store import ChromaStore
from app.utils import extract_pdf_text, compute_file_hash, create_slug, chunk_text


@click.group()
def cli():
    """Local Multimodal AI Agent - Intelligent document and image management."""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--topics', required=True, help='Comma-separated list of topics (e.g., "CV,NLP,RL")')
@click.option('--copy', is_flag=True, help='Keep original file (copy instead of move)')
def add_paper(pdf_path: str, topics: str, copy: bool):
    """Add and classify a PDF paper based on specified topics."""
    click.echo(f"📄 Processing paper: {pdf_path}")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.suffix.lower() == '.pdf':
        click.echo("❌ Error: File must be a PDF", err=True)
        return
    
    # Parse topics
    topic_list = [t.strip() for t in topics.split(',')]
    click.echo(f"📚 Available topics: {', '.join(topic_list)}")
    
    # Extract text from PDF
    click.echo("🔍 Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        click.echo("⚠️  Warning: Could not extract text from PDF")
        return
    
    # Initialize embedder and store
    embedder = TextEmbedder()
    store = ChromaStore()
    
    # Classify paper
    click.echo("🤖 Classifying paper using AI...")
    best_topic = classify_paper(text, topic_list, embedder)
    click.echo(f"✅ Classification result: {best_topic}")
    
    # Create target directory
    target_dir = PAPERS_DIR / best_topic
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Move or copy file
    target_path = target_dir / pdf_path.name
    if copy:
        import shutil
        shutil.copy2(pdf_path, target_path)
        click.echo(f"📋 Copied to: {target_path}")
    else:
        pdf_path.rename(target_path)
        click.echo(f"📁 Moved to: {target_path}")
    
    # Add to vector store
    click.echo("💾 Adding to vector database...")
    file_hash = compute_file_hash(target_path)
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = embedder.embed(chunk)
        metadata = {
            'filename': target_path.name,
            'topic': best_topic,
            'chunk_id': i,
            'total_chunks': len(chunks),
            'file_hash': file_hash,
            'path': str(target_path)
        }
        store.add_document(
            collection_name='papers',
            doc_id=f"{file_hash}_{i}",
            embedding=embedding,
            text=chunk,
            metadata=metadata
        )
    
    click.echo(f"✨ Successfully added paper with {len(chunks)} chunks")


@cli.command()
@click.argument('folder_path', type=click.Path(exists=True))
@click.option('--topics', required=True, help='Comma-separated list of topics')
def organize_papers(folder_path: str, topics: str):
    """Batch organize PDFs in a folder into topic-based subfolders."""
    folder_path = Path(folder_path)
    topic_list = [t.strip() for t in topics.split(',')]
    
    pdf_files = list(folder_path.glob('*.pdf'))
    if not pdf_files:
        click.echo("❌ No PDF files found in folder")
        return
    
    click.echo(f"📚 Found {len(pdf_files)} PDF files to organize")
    click.echo(f"📂 Topics: {', '.join(topic_list)}")
    
    embedder = TextEmbedder()
    store = ChromaStore()
    
    with click.progressbar(pdf_files, label='Processing papers') as bar:
        for pdf_path in bar:
            try:
                text = extract_pdf_text(pdf_path)
                if not text.strip():
                    continue
                
                best_topic = classify_paper(text, topic_list, embedder)
                
                # Create target directory
                target_dir = PAPERS_DIR / best_topic
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Move file
                target_path = target_dir / pdf_path.name
                pdf_path.rename(target_path)
                
                # Add to vector store
                file_hash = compute_file_hash(target_path)
                chunks = chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    embedding = embedder.embed(chunk)
                    metadata = {
                        'filename': target_path.name,
                        'topic': best_topic,
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'file_hash': file_hash,
                        'path': str(target_path)
                    }
                    store.add_document(
                        collection_name='papers',
                        doc_id=f"{file_hash}_{i}",
                        embedding=embedding,
                        text=chunk,
                        metadata=metadata
                    )
            except Exception as e:
                click.echo(f"\n⚠️  Error processing {pdf_path.name}: {str(e)}")
    
    click.echo("✨ Batch organization complete!")


@cli.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
def search_paper(query: str, top_k: int):
    """Search papers using natural language query."""
    click.echo(f"🔍 Searching for: {query}")
    
    embedder = TextEmbedder()
    store = ChromaStore()
    
    # Generate query embedding
    query_embedding = embedder.embed(query)
    
    # Search in vector store
    results = store.search(
        collection_name='papers',
        query_embedding=query_embedding,
        n_results=top_k
    )
    
    if not results:
        click.echo("❌ No results found")
        return
    
    click.echo(f"\n📊 Found {len(results)} relevant results:\n")
    
    seen_files = set()
    for i, result in enumerate(results, 1):
        filename = result['metadata']['filename']
        if filename in seen_files:
            continue
        seen_files.add(filename)
        
        topic = result['metadata'].get('topic', 'Unknown')
        path = result['metadata'].get('path', 'Unknown')
        distance = result.get('distance', 0)
        similarity = 1 - distance
        
        click.echo(f"{i}. {filename}")
        click.echo(f"   📁 Topic: {topic}")
        click.echo(f"   📍 Path: {path}")
        click.echo(f"   🎯 Similarity: {similarity:.2%}")
        click.echo(f"   📝 Snippet: {result['text'][:150]}...")
        click.echo()


@cli.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of images to return')
def search_image(query: str, top_k: int):
    """Search images using natural language description."""
    click.echo(f"🔍 Searching images for: {query}")
    
    # Check if images directory exists and has images
    if not IMAGES_DIR.exists():
        click.echo("❌ Images directory not found. Creating it...")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Please place images in: {IMAGES_DIR}")
        return
    
    image_files = list(IMAGES_DIR.glob('*.[jp][pn]g')) + list(IMAGES_DIR.glob('*.jpeg'))
    if not image_files:
        click.echo(f"❌ No images found in {IMAGES_DIR}")
        return
    
    click.echo(f"📸 Found {len(image_files)} images")
    
    # Initialize image embedder
    image_embedder = ImageEmbedder()
    store = ChromaStore()
    
    # Index images if not already indexed
    click.echo("🤖 Indexing images...")
    for img_path in image_files:
        try:
            file_hash = compute_file_hash(img_path)
            
            # Check if already indexed
            existing = store.get_by_id('images', file_hash)
            if existing:
                continue
            
            # Generate embedding
            embedding = image_embedder.embed_image(img_path)
            metadata = {
                'filename': img_path.name,
                'path': str(img_path),
                'file_hash': file_hash
            }
            
            store.add_document(
                collection_name='images',
                doc_id=file_hash,
                embedding=embedding,
                text=img_path.name,
                metadata=metadata
            )
        except Exception as e:
            click.echo(f"⚠️  Error indexing {img_path.name}: {str(e)}")
    
    # Generate text embedding for query
    text_embedding = image_embedder.embed_text(query)
    
    # Search
    results = store.search(
        collection_name='images',
        query_embedding=text_embedding,
        n_results=top_k
    )
    
    if not results:
        click.echo("❌ No matching images found")
        return
    
    click.echo(f"\n🖼️  Top {len(results)} matching images:\n")
    
    for i, result in enumerate(results, 1):
        filename = result['metadata']['filename']
        path = result['metadata']['path']
        distance = result.get('distance', 0)
        similarity = 1 - distance
        
        click.echo(f"{i}. {filename}")
        click.echo(f"   📍 Path: {path}")
        click.echo(f"   🎯 Similarity: {similarity:.2%}")
        click.echo()


def classify_paper(text: str, topics: List[str], embedder: TextEmbedder) -> str:
    """Classify paper into one of the given topics."""
    # Take first 2000 characters for classification
    text_sample = text[:2000]
    
    # Generate embedding for the paper
    paper_embedding = embedder.embed(text_sample)
    
    # Generate embeddings for each topic
    topic_embeddings = {}
    for topic in topics:
        topic_prompt = f"This is a research paper about {topic}"
        topic_embeddings[topic] = embedder.embed(topic_prompt)
    
    # Compute cosine similarity
    import numpy as np
    
    best_topic = None
    best_similarity = -1
    
    for topic, topic_emb in topic_embeddings.items():
        similarity = np.dot(paper_embedding, topic_emb) / (
            np.linalg.norm(paper_embedding) * np.linalg.norm(topic_emb)
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_topic = topic
    
    return best_topic


if __name__ == '__main__':
    cli()