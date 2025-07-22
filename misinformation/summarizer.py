"""This module provides a class to summarize a piece of text"""

from transformers import pipeline


class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        Summarizes a piece of text
        Args:
            text (str): The text to summarize
            max_length (int): The maximum length of the summary

        Returns:
            str: The summarization
        """
        return self.summarizer(text, max_length=max_length)[0]["summary_text"]