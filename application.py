import utils

from nlp import preprocess, Similarity, clean


import streamlit as st 
import numpy as np
import pandas as pd 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st 
import spacy
# import spacy_streamlit

from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
from collections import Counter
from gensim.summarization import keywords


#from sentence_transformers import SentenceTransformer, util

import plotly.express as px
import plotly.figure_factory as ff

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        footer:after {
        content:'JAMS assist, 2021; Made with Streamlit';
        visibility: visible;
        display: block;
        position: relative;
        text-align:center;
        color:#4A6AD0;
        #background-color: #CBD1C3; 
        padding: 5px;
        top: 2px; }
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


DEF_JOB = "Paste your job description here."
DEF_RESUME = "Paste your resume here."
LIST = ['Home Brew', 'Sklearn', 'Spacy_Builtin']
st.sidebar.title('Similarity Check')
event = st.sidebar.radio('Experiments', LIST)
st.sidebar.subheader('Job Description')
top10 = st.sidebar.checkbox('Display Top 10 Keywords')
st.sidebar.subheader('Resume')
matched = st.sidebar.checkbox('Display Matched Keywords')
st.sidebar.empty()

st.image('./jams.png')

job = st.text_area('Job Description', DEF_JOB, height=400)
resume = st.text_area('Resume', DEF_RESUME, height = 400)

### 'Home Brew Cosine'
pp_job = preprocess(job)
pp_resume = preprocess(resume)

jkw=pd.DataFrame(pp_job.items(), columns=['Terms', 'Frequency'])

jkw = jkw[jkw['Frequency']>=2]
jkw = jkw.sort_values(by='Frequency', ascending=False)
# st.subheader('Top 10 keywords in the Job Description')

# st.dataframe(jkw.head(10))

fig = px.bar(jkw,
                x='Terms',
                y='Frequency',
#                 hover_name='name',
                title='Top 10 Keywords in the Job Description')
if top10:
    st.plotly_chart(fig)
    
    ### Cosine Similarity
sk_job = clean(job)
sk_resume = clean(resume)
text_list = [sk_job, sk_resume]

@st.cache(allow_output_mutation=True)
def process_text(text):
    nlp=spacy.load('en_core_web_sm')
    return nlp(text)
    
### Spacy built in Similarity
doc1 = process_text(sk_job)
doc2 = process_text(sk_resume)


nlpm = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlpm.vocab)
terms = keywords(sk_job, ratio=.5).split('\n')
patterns = [nlpm.make_doc(text) for text in terms]
matcher.add("Spec", patterns)

doc = nlpm(sk_resume)
matchkeywords =[]
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    if len(span.text)>3:
        matchkeywords.append(span.text)
        
a = Counter(matchkeywords)

data = []
for t in terms:
    rec={}
    rec['Terms'] = t
    try:
        rec['Frequency'] = a[t]
    except:
        rec['Frequency'] = 0

    data.append(rec) 
    
# data =data[data['Frequency'] > 2]
# data = data.sort_values(by='Frequency', ascending=False)


# %%
df = pd.DataFrame(data)
df = df[df['Frequency']>=1]
df = df.sort_values(by='Frequency', ascending=False)
# df=df.sort_values(by='Frequency', ascending=False)[:5]
# st.subheader('Matched Keywords')
# st.dataframe(df)

fig1= px.bar(df,
                x='Terms',
                y='Frequency',
#                 hover_name='name',
                title='Matched Keywords')
if matched:
    st.plotly_chart(fig1)

if event == 'Home Brew':
    st.sidebar.markdown("""[Cosine Similarity](https://studymachinelearning.com/cosine-similarity-text-similarity-metric/#:~:text=Cosine%20similarity%20is%20one%20of,size%20in%20Natural%20language%20Processing.&text=If%20the%20Cosine%20similarity%20score,two%20documents%20have%20less%20similarity.)""")
    matchPercentage = np.round((Similarity(pp_job, pp_resume)*100),2)
    if len(job) != len(DEF_JOB) and len(resume) != len(DEF_RESUME):
        st.sidebar.markdown(
        f'<div style="color: green; font-size: largest"> Your resume matched <h1> {matchPercentage}% </h1> with the job description. </h1></div>',
        unsafe_allow_html=True)
#         st.write("Your Resume matched", matchPercentage, '% with the job description')
        if matchPercentage >= 80:
            st.balloons()
            
elif event == 'Sklearn':
    st.sidebar.markdown("""
    *[Sklearn Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)  
    *[Sklearn Example](https://clay-atlas.com/us/blog/2020/03/27/cosine-similarity-text-calculate-python/)""")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage_sk=cosine_similarity(count_matrix)[0][1]*100
    matchPercentage_sk= round(matchPercentage_sk, 2)
    if len(job) != len(DEF_JOB) and len(resume) != len(DEF_RESUME):
        st.sidebar.markdown(
        f'<div style="color: green; font-size: largest"> Your resume matched <h1> {matchPercentage_sk}% </h1> with the job description. </h1></div>',
        unsafe_allow_html=True)
#         st.write("Your Resume matched", matchPercentage_sk, '% with the job description')
        if matchPercentage_sk >= 80:
            st.balloons()
            
elif event == 'Spacy_Builtin':
    st.sidebar.markdown("""[Spacy Vectors and Similarity](https://spacy.io/usage/linguistic-features#vectors-similarity)""")
    matchPercentage_spacy= round(doc1.similarity(doc2),2)*100
    
    if len(job) != len(DEF_JOB) and len(resume) != len(DEF_RESUME):
        st.sidebar.markdown(
        f'<div style="color: green; font-size: largest"> Your resume matched <h1> {matchPercentage_spacy}% </h1> with the job description. </h1></div>',
        unsafe_allow_html=True)
#         st.write("Your Resume matched", matchPercentage_sk, '% with the job description')
        if matchPercentage_spacy >= 80:
            st.balloons()


