from sentence_transformers import SentenceTransformer

def sb_vectorizer(inp_sentences:list):
    '''
        transforms a list of string into a vector representation with sentenceBert
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(inp_sentences)

    return embeddings