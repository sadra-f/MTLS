#====================================================================================
#set the dir for the Read/Write files and classes * Note to follow the pattern use dir or file name alike the original pattern for log files
#====================================================================================



_STR_INPUT_PATH = 'test_io/input.txt'
_STR_LOG_PATH = 'log/.log'

_STR_DATASET_PATH = 'dataset/mtl_dataset/L5/D3'
_STR_READY_HT_PATH = 'C:/Users/TOP/Desktop/project/mtl_dataset_HT/mtl_dataset/L5/D3'

_STR_SENTENCE_BERT_VECTORS_PATH = 'log/SENTENCE_BERT_VECTORS/sb.npy'
_STR_SENT_LIST_PATH = 'log/sentences/sentences.npy'
_STR_SENT_HT_LIST_PATH = 'log/sentences/sentences_HT.npy'

_STR_CLUSTER1_DIST_PATH = 'log/dist1/dist1.npy'
_STR_CLUSTER1_SORTED_DIST_PATH = 'log/sorted_dist1/sortd_dist1.npy'
_STR_CLUSTER1_RES_PATH = 'log/cluster1_res/cluster1.npy'

_STR_CLUSTER2_DIST_PATH = 'log/dist2/'
_STR_CLUSTER2_RES_PATH = 'log/cluster2_res/cluster2.npy'
#====================================================================================

from pathlib import Path

INPUT_PATH = Path(_STR_INPUT_PATH)
LOG_PATH = Path(_STR_LOG_PATH)

DATASET_PATH = Path(_STR_DATASET_PATH)
READY_HT_PATH = Path (_STR_READY_HT_PATH)

SENTENCE_BERT_VECTORS_PATH = Path (_STR_SENTENCE_BERT_VECTORS_PATH)
SENT_LIST_PATH = Path (_STR_SENT_LIST_PATH)
SENT_HT_LIST_PATH = Path (_STR_SENT_HT_LIST_PATH)

CLUSTER1_DIST_PATH = Path (_STR_CLUSTER1_DIST_PATH)
CLUSTER1_RES_PATH = Path (_STR_CLUSTER1_RES_PATH)
CLUSTER1_SORTED_DIST_PATH = Path (_STR_CLUSTER1_SORTED_DIST_PATH)


CLUSTER2_DIST_PATH = Path (_STR_CLUSTER2_DIST_PATH)
CLUSTER2_RES_PATH = Path(_STR_CLUSTER2_RES_PATH)