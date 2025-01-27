from typing import List
from rouge_score import rouge_scorer

class ROUGEMetric:
    def __init__(self, metric: str):
        """
        Initialize the ROUGEMetric class with the selected ROUGE metric.
        :param metric: The ROUGE metric to calculate ('rouge1', 'rouge2', 'rougeL').
        """
        if metric not in ["rouge1", "rouge2", "rougeL"]:
            raise ValueError("Invalid metric. Choose from 'rouge1', 'rouge2', or 'rougeL'.")
        self.metric = metric
        self.name = "ROUGE-" + metric
        self.scorer = rouge_scorer.RougeScorer([self.metric], use_stemmer=True)

    def calculate_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate the selected ROUGE score for a single reference and hypothesis.
        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The selected ROUGE score as a float.
        """
        score = self.scorer.score(reference, hypothesis)
        return score[self.metric].fmeasure

    def batch_similarity(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Calculate the selected ROUGE score for multiple pairs of references and hypotheses.
        :param references: List of reference texts.
        :param hypotheses: List of hypothesis texts.
        :return: List of selected ROUGE scores as floats.
        """
        if len(references) != len(hypotheses):
            raise ValueError("Input lists must have the same length.")

        results = []
        for ref, hyp in zip(references, hypotheses):
            results.append(self.calculate_similarity(ref, hyp))
        return results
