import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os

def create_ecommerce_knowledge_base():
    print("Loading dataset...")

    # Load the dataset
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    print(f"Dataset loaded: {len(dataset['train'])} examples")

    # Prepare knowledge base
    knowledge_base = []
    for example in dataset['train']:
        # Clean the response
        response = example['response']
        response = response.replace("{{Order Number}}", "your order")
        response = response.replace("{{Online Company Portal Info}}", "our website")
        response = response.replace("{{Online Order Interaction}}", "Order History")
        response = response.replace("{{function replace: Any Hours}}", "business hours")
        response = response.replace("{{Customer Support Phone Number}}", "our support line")
        response = response.replace("{{Website URL}}", "our website")

        # Add to knowledge base
        knowledge_base.append({
            'question': example['instruction'],
            'answer': response,
            'intent': example['intent'],
            'category': example['category']
        })

    print(f"Knowledge base created with {len(knowledge_base)} entries")

    # Load sentence transformer for embeddings
    print("Loading sentence transformer model...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # Create embeddings for all questions
    print("Creating embeddings...")
    questions = [item['question'] for item in knowledge_base]
    embeddings = model.encode(questions, show_progress_bar=True)

    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    # Save knowledge base
    with open('knowledge_base.pkl', 'wb') as f:
        pickle.dump(knowledge_base, f)

    # Save FAISS index
    faiss.write_index(index, 'ecommerce_index.faiss')

    # Save model name for later loading
    with open('model_name.txt', 'w') as f:
        f.write(model_name)

    print("\n Knowledge base created successfully!")
    print("Files created:")
    print("- knowledge_base.pkl")
    print("- ecommerce_index.faiss")
    print("- model_name.txt")

if __name__ == "__main__":
    create_ecommerce_knowledge_base()
