


import utils

from nlp import preprocess, Similarity, clean

# from spacy_similarity import spacy_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn_cosine import sk_cos_similarity
import streamlit as st 
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy
import spacy_streamlit

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    


def_job = "Paste your job description here."
def_resume = "Paste your resume here."

st.sidebar.title('Which Similarity')
select_similarity = st.sidebar.selectbox('Select one',
                                         ['Home Brew Cosine', 'Spacy Similairty','Sklearn Cosine', 'BERT Cosine'])
st.title('Resume Match')

job = st.text_area('Paste your Job Description', def_job)
resume = st.text_area('Paste your Resume', def_resume)

nlp=spacy.load('en_core_web_md')

doc1 = nlp(job)
doc2 = nlp(resume)


### 'Home Brew Cosine'

pp_job = preprocess(job)
pp_resume = preprocess(resume)


matchPercentage = np.round((Similarity(pp_job, pp_resume)*100),2)


### Cosine Similarity
    
sk_job = clean(job)
sk_resume = clean(resume)

text_list = [sk_job, sk_resume]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text_list)
matchPercentage_sk=cosine_similarity(count_matrix)[0][1]*100
matchPercentage_sk= round(matchPercentage_sk, 2)

### Spacy built in Similarity
matchPercentage_spacy= round(doc1.similarity(doc2),2)*100


if select_similarity == "Home Brew Cosine":
    if len(job) != 0 and len(resume) != 0:
        st.write("Your Resume matched", matchPercentage, '% with the job description')    
if select_similarity == "Sklearn Cosine":
    if len(job) != 0 and len(resume) != 0:
        st.write("Your Resume matched", matchPercentage_sk, '% with the job description')
if select_similarity=='Spacy Similarity':
    
    if len(job) != 0 and len(resume) != 0:
        st.write("Your Resume matched", matchPercentage_spacy, '% with the job description')
       
