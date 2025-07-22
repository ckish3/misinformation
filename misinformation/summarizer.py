"""This module provides a class to summarize a piece of text"""

from transformers import pipeline
from rouge_score import rouge_scorer


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

    def evaluate_rouge(self, text: str, reference: str) -> dict:
        """
        Evaluates the ROUGE score of a piece of text
        Args:
            text (str): The text to evaluate
            reference (str): The reference text

        Returns:
            dict: The ROUGE scores as a dictionary, where the keys are the metric names
                and the values are a Score object
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores
