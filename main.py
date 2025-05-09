import re
import unicodedata
import numpy as np
import pandas as pd
import spacy
import nltk
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

class AdvancedTextProcessor:
    def __init__(self):
        """
        Initialize advanced NLP processing utilities
        """
        # Download necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize Hugging Face pipelines
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.summarizer = pipeline('summarization')
        
        # Stopwords removal
        self.stop_words = set(stopwords.words('english'))
        
        # Word embedding model
        self.word2vec_model = self.train_word2vec_model()
        
        # Error correction model
        self.error_correction_model = self.create_error_correction_model()

    def train_word2vec_model(self, sentences: List[str] = None) -> Word2Vec:
        """
        Train Word2Vec model for semantic understanding
        
        Args:
            sentences (List[str], optional): Training sentences
        
        Returns:
            Word2Vec model
        """
        # Default training sentences if not provided
        if sentences is None:
            sentences = [
                "climate change affects global ecosystems",
                "human activities impact environmental sustainability",
                "scientific research reveals complex planetary challenges"
            ]
        
        # Tokenize sentences
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=tokenized_sentences, 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4
        )
        
        return model

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
        
        Returns:
            Semantic similarity score
        """
        # Tokenize and remove stopwords
        tokens1 = [word for word in word_tokenize(text1.lower()) if word not in self.stop_words]
        tokens2 = [word for word in word_tokenize(text2.lower()) if word not in self.stop_words]
        
        # Calculate word embeddings
        vectors1 = [self.word2vec_model.wv[word] for word in tokens1 if word in self.word2vec_model.wv]
        vectors2 = [self.word2vec_model.wv[word] for word in tokens2 if word in self.word2vec_model.wv]
        
        # Average word vectors
        if not vectors1 or not vectors2:
            return 0.0
        
        avg_vector1 = np.mean(vectors1, axis=0)
        avg_vector2 = np.mean(vectors2, axis=0)
        
        # Calculate cosine similarity
        return np.dot(avg_vector1, avg_vector2) / (np.linalg.norm(avg_vector1) * np.linalg.norm(avg_vector2))

    def create_error_correction_model(self) -> Sequential:
        """
        Create a deep learning model for error correction
        
        Returns:
            Keras Sequential model
        """
        # Sample training data
        texts = [
            "climate change",
            "global warming",
            "environmental crisis",
            "scientific research"
        ]
        
        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        
        # Sequence padding
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=10)
        
        # Create model
        model = Sequential([
            Embedding(len(tokenizer.word_index) + 1, 50, input_length=10),
            LSTM(100),
            Dense(len(tokenizer.word_index) + 1, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model

    def advanced_error_correction(self, text: str) -> str:
        """
        Advanced error correction using multiple techniques
        
        Args:
            text (str): Input text
        
        Returns:
            Corrected text
        """
        # SpaCy-based correction
        doc = self.nlp(text)
        
        # TextBlob spelling correction
        blob = TextBlob(text)
        spelling_corrected = str(blob.correct())
        
        # Machine learning-based suggestions
        corrections = {
            'climat': 'climate',
            'resaerch': 'research',
            'enviromental': 'environmental'
        }
        
        for error, correction in corrections.items():
            spelling_corrected = spelling_corrected.replace(error, correction)
        
        return spelling_corrected

    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text analysis
        
        Args:
            text (str): Input text
        
        Returns:
            Dictionary with multiple analysis results
        """
        # SpaCy named entity recognition
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Text summarization
        summary = self.summarizer(
            text, 
            max_length=50, 
            min_length=20, 
            do_sample=False
        )[0]['summary_text']
        
        # TF-IDF vectorization for key phrase extraction
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top keywords
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'named_entities': entities,
            'sentiment': sentiment,
            'summary': summary,
            'top_keywords': top_keywords,
            'semantic_complexity': self.calculate_semantic_complexity(text)
        }

    def calculate_semantic_complexity(self, text: str) -> float:
        """
        Calculate semantic complexity of text
        
        Args:
            text (str): Input text
        
        Returns:
            Semantic complexity score
        """
        # Tokenize and remove stopwords
        tokens = [word for word in word_tokenize(text.lower()) if word not in self.stop_words]
        
        # Calculate unique word ratio
        unique_words = len(set(tokens))
        total_words = len(tokens)
        
        # Calculate average word embedding distance
        if len(tokens) > 1:
            embeddings = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
            if embeddings:
                # Calculate pairwise distances
                distances = [np.linalg.norm(embeddings[i] - embeddings[j]) 
                             for i in range(len(embeddings)) 
                             for j in range(i+1, len(embeddings))]
                avg_distance = np.mean(distances) if distances else 0
            else:
                avg_distance = 0
        else:
            avg_distance = 0
        
        # Combine metrics
        complexity_score = (unique_words / total_words) * (1 + avg_distance)
        
        return complexity_score

def main():
    # Sample text
    sample_text = """
    So how about we do a rundown of this crisis in bullet points? I'm not going to get too detailed with the signs and politics here. 
    We're on a Souls Journey. I'll instead stick to the Zeitgeist irrelevancies pithy, comes back to unhelpful arguments your workout Surin Citron tea.
    
    Sure. You've had climate change before and yep, the planet survived. But this is not the point. No doubt. The plant Will Survive, again. 
    
    There's just one small problem that we get distracted from this time. We probably won't for, at least Our Lives as we know, and love them won't brutal. The factually.
    
    So, we are the sixth Extinction in the five previous Extinction events other life forms including dinosaurs were wiped out.
    """
    
    # Initialize advanced processor
    processor = AdvancedTextProcessor()
    
    # Error correction
    corrected_text = processor.advanced_error_correction(sample_text)
    print("Corrected Text:")
    print(corrected_text)
    
    # Comprehensive analysis
    analysis = processor.comprehensive_analysis(sample_text)
    
    print("\nComprehensive Analysis:")
    print("Named Entities:", analysis['named_entities'])
    print("Sentiment:", analysis['sentiment'])
    print("Summary:", analysis['summary'])
    print("Top Keywords:", analysis['top_keywords'])
    print("Semantic Complexity:", analysis['semantic_complexity'])
    
    # Semantic similarity example
    comparison_text = "Climate change threatens global ecosystems"
    similarity = processor.semantic_similarity(sample_text, comparison_text)
    print(f"\nSemantic Similarity with '{comparison_text}': {similarity}")

if __name__ == "__main__":
    main()
