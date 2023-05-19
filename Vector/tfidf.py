from sklearn.feature_extraction import text as skt

def tfidf(doc):
    # runs tfdif on a list of strings (a document)
    tfidf = skt.TfidfVectorizer(input='content', smooth_idf=True, norm='l2')
    return (tfidf.fit_transform(doc), tfidf.get_feature_names_out())

def tfidf_list(doc_list):
    #runs tfidf on a list of list of strings (multiple documents)
    res_list = []
    for i in range(len(doc_list)):
        res_list.append(tfidf(doc_list[i]))

    return res_list