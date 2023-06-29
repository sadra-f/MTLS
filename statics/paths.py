#====================================================================================
#set the dir for the Read/Write files and classes * Note to follow the pattern use dir or file name alike the original pattern for log files
#====================================================================================



STR_INPUT_PATH = 'test_io/input.txt'
STR_DATASET_PATH = 'dataset/mtl_dataset/L2/D3'
STR_SENT_LIST_PATH= 'log/SENTENCES/'
STR_HT_LOG_PATH = 'C:/Users/TOP/Desktop/project/mtl_dataset_HT/mtl_dataset/L2/D3'
STR_SENTENCE_BERT_VECTORS_PATH = 'log/SENTENCE_BERT_VECTORS/'
STR_CLUSTER1_DIST_PATH = 'log/DIST1/'
STR_SENTENCE_CLUSTERS_PATH = 'log/CLUSTERS1/'
STR_CLUSTERED_SENTENCES_PATH = 'log/CLUSTERED_SENTENCES/'
STR_CLUSTER_MAIN_PHRASES_PATH = 'log/CLUSTER_MAIN_PHRASES/'
STR_CLUSTER_DOC_VECTOR_PATH = 'log/CLUSTER_DOC_VECTOR/'

#====================================================================================

from pathlib import Path

INPUT_PATH = Path(STR_INPUT_PATH)
DATASET_PATH = Path(STR_DATASET_PATH)

SENT_LIST_PATH = Path (STR_SENT_LIST_PATH)
HT_LOG_PATH = Path (STR_HT_LOG_PATH)
SENTENCE_BERT_VECTORS_PATH = Path (STR_SENTENCE_BERT_VECTORS_PATH)
CLUSTER1_DIST_PATH = Path (STR_CLUSTER1_DIST_PATH)
SENTENCE_CLUSTERS_PATH = Path (STR_SENTENCE_CLUSTERS_PATH)
CLUSTERED_SENTENCES_PATH = Path (STR_CLUSTERED_SENTENCES_PATH)
CLUSTER_MAIN_PHRASES_PATH = Path (STR_CLUSTER_MAIN_PHRASES_PATH)
CLUSTER_DOC_VECTOR_PATH = Path (STR_CLUSTER_DOC_VECTOR_PATH)

