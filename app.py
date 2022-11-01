import pandas as pd
import numpy as np
import streamlit as st
import pickle 

loaded_model=pickle.load(open("classification_model (2).sav","rb"))

def loan_approval(input_data):
    input_data=np.asarray(input_data)
    input_data_reshape=input_data.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    print(prediction)
    if(prediction[0]==0):
        return "Loan not approved"
    else:
        return "Loan is approved"



def main():
    st.title('Loan approval Predictor app')

    Gender=st.text_input('Gender')
    Marital_Status=st.text_input('Married')
    Dependents=st.text_input('Dependents')
    Education=st.text_input('Education')
    Self_Employed=st.text_input('Self_Employed')
    Applicant_Income=st.text_input('ApplicantIncome')
    Coapplicants_income=st.text_input('CoapplicantIncome')
    Loan_Amount=st.text_input('LoanAmount')
    Term=st.text_input('Loan_Amount_Term')
    Credit_History=st.text_input('Credit_History')
    Property_Area=st.text_input('Property_Area')

    loan_prediction=" "

    if st.button('Loan_approval'):
        loan_prediction=loan_prediction([Gender,Marital_Status,Dependents,Education,Self_Employed,Applicant_Income,Coapplicants_income,Loan_Amount,Term,Credit_History,Property_Area])

    st.success(loan_prediction)

if __name__ == '__main__':
    main()
