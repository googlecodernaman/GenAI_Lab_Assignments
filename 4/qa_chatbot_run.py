#!/usr/bin/env python3
"""
Question-Answering Chatbot Implementation
Fixes the Keras compatibility issue by explicitly using PyTorch framework
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Question-Answering Chatbot Implementation")
print("="*80)

# Load pre-trained question-answering model with PyTorch framework
print("\n[1/5] Loading QA model (distilbert-base-cased-distilled-squad)...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt")
print("✓ Model loaded successfully!")

# Define context
print("\n[2/5] Loading context...")
context = """
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
especially computer systems. These processes include learning, reasoning, and self-correction. 
Machine Learning is a subset of AI that provides systems the ability to automatically learn 
and improve from experience without being explicitly programmed. Deep Learning is a subset 
of machine learning that uses neural networks with multiple layers. The field of AI was 
founded in 1956 at a conference at Dartmouth College. AI applications include expert systems, 
natural language processing, speech recognition, and computer vision. Python is one of the 
most popular programming languages for AI and machine learning development. TensorFlow and 
PyTorch are popular frameworks for building deep learning models. Neural networks are inspired 
by the structure and function of the human brain.
"""
print(f"✓ Context loaded! ({len(context)} characters)")

# Test with sample questions
print("\n[3/5] Testing with sample questions...")
test_questions = [
    "What is Artificial Intelligence?",
    "When was AI founded?",
    "What programming language is popular for AI?",
    "What are some AI frameworks?",
    "What is Machine Learning?"
]

print("\n" + "="*80)
print("CHATBOT TEST RESULTS")
print("="*80)

for question in test_questions:
    result = qa_pipeline(question=question, context=context)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"   Confidence: {result['score']:.4f}")
    print("-"*80)

# Define chatbot function
print("\n[4/5] Creating chatbot function...")

def qa_chatbot(question, context, min_confidence=0.01):
    """
    Simple QA chatbot function
    
    Args:
        question: User's question
        context: Context/knowledge base to answer from
        min_confidence: Minimum confidence threshold (default: 0.01)
    
    Returns:
        Dictionary with answer and confidence score
    """
    result = qa_pipeline(question=question, context=context)
    
    if result['score'] >= min_confidence:
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'status': 'success'
        }
    else:
        return {
            'answer': "I'm not confident enough to answer this question based on the given context.",
            'confidence': result['score'],
            'status': 'low_confidence'
        }

print("✓ Chatbot function created!")

# Test the function
print("\n[5/5] Testing chatbot function...")
response = qa_chatbot("What is Deep Learning?", context)
print(f"\nExample: What is Deep Learning?")
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.4f}")

print("\n" + "="*80)
print("✓ CHATBOT IMPLEMENTATION COMPLETE!")
print("="*80)
print("\nThe QA chatbot is now ready to use!")
print("You can call qa_chatbot(question, context) to get answers.")
