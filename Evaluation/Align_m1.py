
import numpy as np
from collections import Counter
from datetime import datetime
from models.TStr import TStr

def rouge_n(system_summary: str, reference_summary: str, n: int) -> float:
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.split()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    system_ngrams = get_ngrams(system_summary, n)
    reference_ngrams = get_ngrams(reference_summary, n)
    
    overlap = sum((system_ngrams & reference_ngrams).values())
    total_reference = sum(reference_ngrams.values())
    
    if total_reference == 0:
        return 0.0
    return overlap / total_reference

def date_distance(d1: datetime, d2: datetime) -> int:
    return abs((d1 - d2).days)


def align_m1_rouge(predictions: list[list[TStr]], gts: list[list[TStr]]) -> float:
    res = []
    for i, val1 in enumerate(predictions):
        for j, val2 in enumerate(gts):
            alignment_scores = []
            for pred_summary in val1:
                best_score = 0
                for gt_summary in val2:
                    date_dist = date_distance(pred_summary.date, gt_summary.date.date())
                    temporal_weight = 1 / (date_dist + 1)
                    content_similarity = rouge_n(pred_summary, gt_summary, 1)
                    alignment_score = temporal_weight * content_similarity
                    best_score = max(best_score, alignment_score)
                alignment_scores.append(best_score)
            
            res.append(np.mean(alignment_scores))
    return res



# # Example usage
# if __name__ == "__main__":
#     predictions = [("2020-01-01", "Event A happened."), ("2020-01-02", "Event B happened.")]
#     gts = [("2020-01-05", "Event A happened."), ("2020-01-02", "Event B happened.")]
#     rouge_score = align_m1_rouge(predictions, gts)
#     print(f"Align+m:1 ROUGE score: {rouge_score}")
