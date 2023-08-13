import numpy as np

def cluster_inp_list(inp_sentence_list, cluster_labels, cluster_count):
    '''
        reformats the initial input list of strings placing strings which are in the 
        same cluster into the same row in a 2d list
    '''
    clustered_sentences = []
    for ci in range(cluster_count):
        clustered_sentences.append([])
        for inpi in np.where(cluster_labels == ci)[0]:
            clustered_sentences[ci].append(inp_sentence_list[inpi])
    
    return clustered_sentences


def dbscan_eps(distances, min_points):
    """
        distances: the distance matrix to be given to the dbscan algorithem to find clusters within
        min_points: the minimum points to be around a point for it to be a centroid

        returns: desirable epsilon value for dbscan algorithem
    """
    return np.sum([float(np.sort(val, axis=0)[min_points]) for val in distances]) / len(distances)
