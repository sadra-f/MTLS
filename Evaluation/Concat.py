import numpy as np
import evaluate as eval

def concat_rouge(timelines, ground_truth):
    TIMELINE_COUNT = len(timelines)

    rouge = eval.load('rouge')
    evaluations = np.ndarray((TIMELINE_COUNT, len(ground_truth)), dtype=object)
    for i in range(TIMELINE_COUNT):
        for j in range(len(ground_truth)):
            evaluation = rouge.compute(predictions=[' '.join(timelines[i])], references=[' '.join(ground_truth[j])])
            evaluations[i][j] = evaluation
    return evaluations