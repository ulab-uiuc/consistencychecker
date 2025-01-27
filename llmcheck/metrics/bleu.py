from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List

class BLEUMetric:
    name: str = "BLEU"
    
    def __init__(self, smoothing_function: bool = True):
        self.smoothing_function = SmoothingFunction().method1 if smoothing_function else None

    def calculate_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate the BLEU score for a single reference and hypothesis.
        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: BLEU score as a float.
        """
        reference_tokens = [reference.split()]
        hypothesis_tokens = hypothesis.split()

        bleu_score = sentence_bleu(
            reference_tokens, hypothesis_tokens, smoothing_function=self.smoothing_function
        )
        return bleu_score

    def batch_similarity(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Calculate BLEU scores for multiple pairs of references and hypotheses.
        :param references: List of reference texts.
        :param hypotheses: List of hypothesis texts.
        :return: List of BLEU scores as floats.
        """
        if len(references) != len(hypotheses):
            raise ValueError("Input lists must have the same length.")

        scores = []
        for ref, hyp in zip(references, hypotheses):
            scores.append(self.calculate_similarity(ref, hyp))
        return scores