""" This module provides functions to split a piece of text into sentences """


import spacy
from typing import List

def get_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """
    Loads a SpaCy model
    Args:
        model_name (str): The name of the SpaCy model to load. Default is "en_core_web_sm"

    Returns:
        spacy.Language: The loaded SpaCy model
    """
    nlp = spacy.load(model_name)
    return nlp


def get_sentences(text: str, nlp: spacy.Language) -> List[str]:
    """
    Splits a piece of text into sentences
    Args:
        text (str): The text to split
        nlp (spacy.Language): The SpaCy model to use

    Returns:
        List[str]: A list of sentences
    """
    doc = nlp(text)

    sentences = [sent.text for sent in doc.sents]

    return sentences