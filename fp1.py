import streamlit as st
import pandas as pd 
import numpy as np

import sklearn as metrics 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

s = pd.read_excel('social_media_usage.xlsx')

def clean_sm(df):
    df = np.where(df == 1, 1, 0)
    return df

ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "age":np.where(s["age"] > 98, np.nan, s["age"]),
    "parent":clean_sm(s["par"]),
    "married":clean_sm(s["marital"]),
    "female":clean_sm(s["gender"]),
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "age", "married", "female", "parent"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     stratify=y,       # same number of target in training & test set,
                                                     test_size=0.2,    # hold out 20% of data for testing,
                                                     random_state=987) # set for reproducibility,
    


# Initialize algorithm 
lr = LogisticRegression(class_weight = "balanced")


# Fit algorithm to training data
lr.fit(X_train, y_train)


st.title('Welcome to my Linkedin User Predictor App!')
st.title('Please select the following :red[Criteria]')

answer_age = st.number_input("What is your age?", value=None, placeholder="Type your Age...")
st.write('Your current age is ', answer_age)

answer_income = st.number_input('What is your income? :money_with_wings:',value=None, placeholder="Type your Income...")
st.write("My income is", answer_income, 'dollars')

answer_education = st.selectbox(
    "What is your education? :book:",
    ("1", "2", "3", "4", "5", "6", "7", "8"),
    index=None,
    placeholder= "Select education level..."
)

st.write('You selected:', answer_education)

answer_parent = st.slider('Are you a parent? :family: 1 for yes and 0 for no', 0, 1)
st.write("I am a  ", answer_parent,)

answer_married = st.slider('Are you married? :ring: 1 for yes and 0 for no', 0, 1)
st.write("I am a  ", answer_married,)

answer_female = st.slider('Are you a female? :female_sign: 1 for yes and 0 for no', 0, 1)
st.write("I am a  ", answer_female)


probs = lr.predict_proba(X_train)
probability = probs[0][1]

if(probability) >=0.5:
    st.write("You are a Linkedin User!")
else:
    st.write("You are not a Linkedin User!")
    
st.write('Probability of Linkedin user is  ', probability)