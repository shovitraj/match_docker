

def sk_cos_similarity(text_list):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage=cosine_similarity(count_matrix)[0][1]*100
    matchPercentage= round(matchPercentage, 2)
    return sk_cos_similarity


