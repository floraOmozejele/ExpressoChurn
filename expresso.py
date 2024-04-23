import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('expresso_processed.csv')

st.markdown("<h1 style = 'color: #FF204E; text-align: center; font-size: 60px; font-family:Helvetica'>EXPRESSO CHURN PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #A0153E; text-align: center; font-family: italic'>BUILT BY FLORA JAMES </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)


# #add image
st.image('IMG_0115.JPG',width = 600)

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("The objective of the espresso churn probability project is to develop a predictive model that can accurately forecast the likelihood of customers churning, or discontinuing their subscription to the espresso service. This model will be based on historical customer data, including factors such as purchase frequency, customer demographics, engagement levels, and any other relevant variables. The goal is to identify patterns and indicators that can help the company proactively take steps to retain at-risk customers and reduce churn, ultimately leading to improved customer retention and increased revenue.")
st.sidebar.image('IMG_0119.JPG', width = 200,caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader('User Input Variables')

sel_cols = ['DATA_VOLUME', 'ON_NET', 'REGULARITY', 'REVENUE', 'FREQUENCE', 'MONTANT',
            'FREQUENCE_RECH', 'ARPU_SEGMENT', 'CHURN']


data_vol = st.sidebar.number_input('DATA_VOLUME', data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
on = st.sidebar.number_input('ON_NET', data['ON_NET'].min(), data['ON_NET'].max())
reg = st.sidebar.number_input('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
rev = st.sidebar.number_input('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
freq = st.sidebar.number_input('FREQUENCE', data['FREQUENCE'].min(), data['FREQUENCE'].max())
mont= st.sidebar.number_input('MONTANT', data['MONTANT'].min(), data['MONTANT'].max())
freq_rec = st.sidebar.number_input('FREQUENCE_RECH', data['FREQUENCE_RECH'].min(), data['FREQUENCE_RECH'].max())
arp = st.sidebar.number_input('ARPU_SEGMENT', data['ARPU_SEGMENT'].min(), data['ARPU_SEGMENT'].max())


#users input
input_var = pd.DataFrame()
input_var['DATA_VOLUME'] = [data_vol]
input_var['ON_NET'] = [on]
input_var['REGULARITY'] = [reg]
input_var['REVENUE'] = [rev]
input_var['FREQUENCE'] = [freq]
input_var['MONTANT'] = [mont]
input_var['FREQUENCE_RECH'] = [freq_rec]
input_var['ARPU_SEGMENT'] = [arp]



st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)


#import the transformers
data_vol = joblib.load('DATA_VOLUME_scaler.pkl')
rev = joblib.load('REVENUE_scaler.pkl')
mont = joblib.load('MONTANT_scaler.pkl')
arp = joblib.load('ARPU_SEGMENT_scaler.pkl')




# transform the users input with the imported encoders
input_var['DATA_VOLUME'] = data_vol.transform(input_var[['DATA_VOLUME']])
input_var['REVENUE'] = rev.transform(input_var[['REVENUE']])
input_var['MONTANT'] = mont.transform(input_var[['MONTANT']])
input_var['ARPU_SEGMENT'] = arp.transform(input_var[['ARPU_SEGMENT']])



st.header('Transformed Input Variable')
st.dataframe(input_var, use_container_width = True)

# st.dataframe(input_var)
model = joblib.load('ExpressorModel.pkl')



predict = model.predict(input_var)

if st.button('Check Churn Probability'):
    predicted_churn= model.predict(input_var)
    st.success(f"Your Company's Probability Churn is { predicted_churn[0].round(2)}")
    st.snow()