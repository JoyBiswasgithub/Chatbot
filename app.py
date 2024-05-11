from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
cont = """Enthusiastic and self-motivated web designer with 5+ years of experience. Eager to join Studio Shodwe and
            bring my skill in frontend development, and visual design to every project that will be received in the
            future. A previous project for improving and redesigning reallygreatsite.com resulted in an increase in
            web traffic by 50% and performance improvement by 20%.
            """
question = 'How many percent inproved the web traffic'
QA_input = {
    
    'question': question,
    'context': cont
    }
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
res = nlp(QA_input)

#print(res)

st.write(res)
