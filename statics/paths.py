#====================================================================================
#set the dir for the Read/Write files and classes * Note to follow the pattern use dir or file name alike the original pattern for log files
#====================================================================================
from .config import N_TIMELINES, DATASET_NUMBER


_STR_DATASET_PATH = f'dataset/mtl_dataset/L{N_TIMELINES}/D{DATASET_NUMBER}'
_STR_READY_HT_PATH = f'../../mtl_dataset_HT/mtl_dataset/L{N_TIMELINES}/D{DATASET_NUMBER}'

_STR_SENTENCE_BERT_VECTORS_PATH = 'log/SENTENCE_BERT_VECTORS/'
_STR_GT_SENTENCE_BERT_VECTORS_PATH = 'log/GT_SENTENCE_BERT_VECTORS/'

_STR_INIT_CLUSTER_LABELS_PATH = 'log/initclusterlabel/'
_STR_INIT_CLUSTER_SENT_PATH = 'log/initclustersents/'
_STR_CLUSTER1_DIST_PATH = 'D:/programing/log_data/bach_prj/dist1/'
_STR_CLUSTER1_CLEAN_SENTENCES_PATH = 'log/cluster1_clean_sent_res/'

_STR_BFNSP_RES_PATH = 'log/bfnsp/'
#====================================================================================

from pathlib import Path


DATASET_PATH = Path(_STR_DATASET_PATH)
READY_HT_PATH = Path (_STR_READY_HT_PATH)

INIT_CLUSTER_LABELS_PATH = Path(_STR_INIT_CLUSTER_LABELS_PATH)
INIT_CLUSTER_SENT_PATH = Path(_STR_INIT_CLUSTER_SENT_PATH)
SENTENCE_BERT_VECTORS_PATH = Path (_STR_SENTENCE_BERT_VECTORS_PATH)
GT_SENTENCE_BERT_VECTORS_PATH = Path(_STR_GT_SENTENCE_BERT_VECTORS_PATH)

CLUSTER1_DIST_PATH = Path (_STR_CLUSTER1_DIST_PATH)

CLUSTER1_CLEAN_SENTENCES_PATH = Path(_STR_CLUSTER1_CLEAN_SENTENCES_PATH)
BFNSP_RES_PATH = Path(_STR_BFNSP_RES_PATH)