import numpy as np
from sklearn.metrics import pairwise_distances
from helpers.distances import cosine_similarity
from models.TStr import TStr
from datetime import datetime


def temporal_weight(d_diff, date_weight = "linear"):
    if date_weight == 'linear':
        return 1 / (d_diff + 1)
    elif date_weight == 'inverse':
        return 1 / (d_diff + 1)
    else:
        raise ValueError("Unsupported date_weight value. Use 'linear' or 'inverse'.")
    
def date_difference(d1, d2):
    return abs((d1 - d2).days)

def calculate_d_select(predictions:list[TStr], gts:list[TStr], date_weight='linear'):
    """
    Calculate the d-Select metric for two sets of sentences and dates.
    
    :param predictions: List of tuples (date, summary) for predicted timelines
    :param gts: List of tuples (date, summary) for ground truth timelines
    :param date_weight: Weighting scheme for date differences. Options: 'linear', 'inverse'.
    :return: d-Select score
    """
      
    res = []
    for i, prd in enumerate(predictions):
        for j, gt in enumerate(gts):
            S = {s.date: (s, s.vector) for s in prd}
            R = {s.date.date(): (s,s.vector) for s in gt}
            
            all_dates_R = list(R.keys())
            all_dates_S = list(S.keys())
            
            # Compute temporal alignment
            aligned_dates = []
            for dr in all_dates_R:
                best_match = None
                best_weight = -1
                for ds in all_dates_S:
                    t_weight = temporal_weight(date_difference(dr, ds), date_weight)
                    if t_weight > best_weight:
                        best_match = ds
                        best_weight = t_weight
                if best_match:
                    aligned_dates.append((dr, best_match))
            
            # Compute d-Select score
            d_select_score = 0
            for dr, ds in aligned_dates:
                sim_score = cosine_similarity(R[dr][1], S[ds][1])
                t_weight = temporal_weight(date_difference(dr, ds))
                d_select_score += t_weight * sim_score
            
            # Normalize the score by the number of ground truth dates
            d_select_score /= len(all_dates_R)
            
            res.append(d_select_score)
    return res

# Example usage
# from datetime import datetime

# # Example data
# predictions = [(datetime(2020, 1, 1), 'Event A'), (datetime(2020, 1, 2), 'Event B')]
# gts = [(datetime(2020, 1, 1), 'Event A'), (datetime(2020, 1, 2), 'Event B')]

# # Calculate d-Select
# d_select = calculate_d_select(predictions, gts)
# print(f"d-Select Score: {d_select}")
