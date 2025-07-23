

import transformers
import torch

class LLM_Evaluator():
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        return self.pipeline(prompt, max_length=max_length)[0]["generated_text"]

    def evaluate(self, document: str, summary: str) -> int:
        """
        Evaluates the consistency (factual accuracy) of a summary of the given document. The result is an integer
        between 1 and 5.

        Args:
            document (str): The full, unsummarized text
            summary (str): The summary of the document

        Returns:
            int: The consistency score
        """
        
        EVALUATION_PROMPT_TEMPLATE = """
        You will be given one summary written for an article. Your task is to rate the summary on one metric.
        Please make sure you read and understand these instructions very carefully. 
        Please keep this document open while reviewing, and refer to it as needed.

        Evaluation Criteria:

        {criteria}

        Evaluation Steps:

        {steps}

        Example:

        Source Text:

        {document}

        Summary:

        {summary}

        Evaluation Form (scores ONLY):

        - {metric_name}
        """

        CONSISTENCY_SCORE_CRITERIA = """
        Consistency(1-5) - the factual alignment between the summary and the summarized source. \
        A factually consistent summary contains only statements that are entailed by the source document. \
        Annotators were also asked to penalize summaries that contained hallucinated facts.
        """

        CONSISTENCY_SCORE_STEPS = """
        1. Read the article carefully and identify the main facts and details it presents.
        2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
        3. Assign a score for consistency based on the Evaluation Criteria.
        """

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            criteria=CONSISTENCY_SCORE_CRITERIA,
            steps=CONSISTENCY_SCORE_STEPS,
            metric_name='Consistency',
            document=document,
            summary=summary,
        )
        response = self.generate_response(prompt)

        score_num = int(response.strip())

        return score_num