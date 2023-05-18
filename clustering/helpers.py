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