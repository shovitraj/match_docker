
import streamlit as st 
import spacy
# import spacy_streamlit

nlp=spacy.load('en_core_web_md')


job = st.text_area("This is a job description.", 'Paste your job description here.')
resume = st.text_area("This is a resume.", 'Paste your resume here')

job = job.lower()
job = job.replace('\n', '')
job = job.replace('_', '')

resume = resume.lower()
resume = resume.replace('\n', '')
resume = resume.replace('_', '')

### Spacy built in Similarity
doc1 = nlp(job)
doc2 = nlp(resume)
matchPercentage_spacy= round(doc1.similarity(doc2),2)*100

st.write("Your Resume matched", matchPercentage_spacy, '% with the job description')
