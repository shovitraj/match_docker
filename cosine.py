
import utils
import streamlit as st 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job = st.text_area("This is a job description.", 'Paste your job description here.')
resume = st.text_area("This is a resume.", 'Paste your resume here')

job = job.lower()
job = job.replace('\n', '')
job = job.replace('_', '')

resume = resume.lower()
resume = resume.replace('\ng', '')
resume = resume.replace('_', '')

text_list = [job, resume]

    
cv = CountVectorizer()
count_matrix = cv.fit_transform(text_list)
matchPercentage=cosine_similarity(count_matrix)[0][1]*100
matchPercentage= round(matchPercentage, 2)

st.write(matchPercentage)
    
