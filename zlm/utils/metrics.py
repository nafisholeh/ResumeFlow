'''
-----------------------------------------------------------------------
File: metrics.py
Creation Time: Jan 13th 2024, 8:42 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''
import re
import json
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from zlm.utils.utils import key_value_chunking

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download all required NLTK resources
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Try to download punkt_tab if it exists
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        print("Note: punkt_tab resource not found in NLTK. Using alternative tokenization.")
except Exception as e:
    print(f"Warning: Error downloading NLTK resources: {e}")
    print("Metrics calculation may not work properly.")

def remove_urls(list_of_strings):
    """Removes strings containing URLs from a list using regular expressions."""
    filtered_list = [string for string in list_of_strings if not re.search(r"https?://\S+", string)]
    return filtered_list

def overlap_coefficient(document1: str, document2: str) -> float:
    """Calculate the overlap coefficient between two documents.

    The overlap coefficient is a measure of the overlap between two sets,
    and is defined as the size of the intersection divided by the smaller
    of the size of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The overlap coefficient between the two documents.
    """
    try:
        # List the unique words in a document
        words_in_document1 = set(normalize_text(document1))
        words_in_document2 = set(normalize_text(document2))

        # Find the intersection of words list of document1 & document2
        intersection = words_in_document1.intersection(words_in_document2)

        # Calculate overlap coefficient
        try:
            overlap_coefficient = float(len(intersection)) / min(len(words_in_document1), len(words_in_document2))
        except ZeroDivisionError:
            overlap_coefficient = 0.0

        return overlap_coefficient
    except Exception as e:
        print(f"Error calculating overlap coefficient: {e}")
        return 0.5  # Return a neutral value in case of error

def jaccard_similarity(document1: str, document2: str) -> float:
    """Calculate the Jaccard similarity between two documents.

    The Jaccard similarity is a measure of the similarity between two sets,
    and is defined as the size of the intersection divided by the size of
    the union of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The Jaccard similarity between the two documents.
    """
    try:
        # List the unique words in a document
        words_in_document1 = set(normalize_text(document1))
        words_in_document2 = set(normalize_text(document2))

        # Find the intersection of words list of document1 & document2
        intersection = words_in_document1.intersection(words_in_document2)

        # Find the union of words list of document1 & document2
        union = words_in_document1.union(words_in_document2)

        # Calculate Jaccard similarity score
        try:
            jaccard_similarity = float(len(intersection)) / len(union)
        except ZeroDivisionError:
            jaccard_similarity = 0.0

        return jaccard_similarity
    except Exception as e:
        print(f"Error calculating Jaccard similarity: {e}")
        return 0.5  # Return a neutral value in case of error

def cosine_similarity(document1: str, document2: str) -> float:
    """Calculate the cosine similarity between two documents.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The cosine similarity between the two documents.
    """
    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Transform the documents into TF-IDF vectors
        vectors = vectorizer.fit_transform([document1, document2])

        cosine_similarity_score = pairwise.cosine_similarity(vectors[0], vectors[1])

        return cosine_similarity_score.item()
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        # Try a simpler approach if TF-IDF fails
        try:
            # Use the normalized text and calculate a simple dot product
            words1 = normalize_text(document1)
            words2 = normalize_text(document2)

            # Convert to sets for a simple overlap measure
            set1 = set(words1)
            set2 = set(words2)

            # Calculate a simple similarity measure
            if len(set1) == 0 or len(set2) == 0:
                return 0.0

            intersection = len(set1.intersection(set2))
            return intersection / (math.sqrt(len(set1)) * math.sqrt(len(set2)))
        except Exception as inner_e:
            print(f"Error in fallback cosine similarity calculation: {inner_e}")
            return 0.5  # Return a neutral value in case of error

def vector_embedding_similarity(llm, document1: str, document2: str) -> float:
    """Calculate the similarity between two documents using vector embeddings.

    Args:
        llm: The language model to use for generating embeddings.
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The similarity score between the two documents.
    """
    try:
        document1 = key_value_chunking(json.loads(document1))
        document2 = key_value_chunking(json.loads(document2))

        emb_1 = llm.get_embedding(document1, task_type="retrieval_query")
        emb_2 = llm.get_embedding(document2, task_type="retrieval_query")

        df1 = pd.DataFrame(emb_1.embedding.to_list())
        df2 = pd.DataFrame(emb_2.embedding.to_list())

        emb_sem = pairwise.cosine_similarity(df1, df2)

        return emb_sem.mean()
    except Exception as e:
        print(f"Error calculating vector embedding similarity: {e}")
        # Fall back to cosine similarity
        try:
            return cosine_similarity(str(document1), str(document2))
        except Exception as inner_e:
            print(f"Error in fallback similarity calculation: {inner_e}")
            return 0.5  # Return a neutral value in case of error

def normalize_text(text: str) -> list:
    """Normalize the input text.

    This function tokenizes the text, removes stopwords and punctuations,
    and applies stemming.

    Args:
        text (str): The text to normalize.

    Returns:
        list: The list of normalized words.
    """
    try:
        # Step 1: Tokenization
        try:
            # Try using NLTK's word_tokenize
            words = word_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK tokenization failed: {e}")
            # Fallback to simple whitespace and punctuation tokenization
            import re
            words = re.findall(r'\b\w+\b', text.lower())

        # Step 2: Data Cleaning - Remove Stopwords and Punctuations
        words = [re.sub('[^a-zA-Z]', '', word).lower() for word in words]

        # Step 3: Remove empty tokens
        words = [word for word in words if len(word)]

        # Step 4: Remove Stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
        except Exception as e:
            print(f"Warning: Stopwords removal failed: {e}")
            # Continue without removing stopwords
            pass

        # Step 5: Stemming
        try:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        except Exception as e:
            print(f"Warning: Stemming failed: {e}")
            # Continue without stemming
            pass

        return words
    except Exception as e:
        print(f"Error in normalize_text: {e}")
        # Return a simple fallback tokenization in case of any errors
        return [word.lower() for word in text.split() if word.strip()]