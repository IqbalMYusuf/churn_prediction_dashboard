# Import required library
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
st.set_page_config(layout="wide")

# Define available dataframe
current_df = pd.read_csv('available_data.csv')
predicted_df = None

# Define function
def hist_plot_facet(data, x, color, facet_col, title):
    fig = px.histogram(data, x=x, color=color, barmode='group', facet_col=facet_col, width=900, height=500)
    fig.update_layout(title={'text':title,
                              'y':1.0,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                      title_font_size=24)
    fig.for_each_xaxis(lambda y: y.update(title_font_size=18, tickfont_size=16))
    fig.for_each_yaxis(lambda y: y.update(title_font_size=18))
    fig.for_each_annotation(lambda y: y.update(text=y.text.replace('gender=', ''), font_size=18))
    return fig

def hist_plot(data, x, title, c_marker='#636EFA'):
    fig = px.histogram(data, x=x, barmode='group', width=900, height=500)
    fig.update_layout(title={'text':title,
                             'font_size':24,
                              'y':1.0,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                      bargap=0.1,
                      yaxis={'tickfont_size':16, 'title_font_size':18},
                      xaxis={'tickfont_size':16, 'title_font_size':18})
    fig.update_traces(marker={'color':c_marker})
    return fig

def pie_plot(data, value, name, title):
    fig = px.pie(data, values=value, names=name, width=600, height=600)
    fig.update_layout(title={ 'text': title,
                              'y':0.97,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                      title_font_size=24)
    fig.add_annotation(dict(font=dict(size=18),
                            x=0.40,
                            y=-0.05,
                            showarrow=False,
                            text="Total: {}".format(sum(data[value])),
                            xanchor='left'))
    fig.update_traces(textfont_size=16)
    return fig

def counts_df(data, columns):
    df = data[columns].value_counts().reset_index()
    df.columns = ['name', 'value']
    return df

# Store current datetime
now = datetime.now()
now_string = now.strftime('%d/%M/%Y %H:%M:%S')

# Streamlit section 1
col1, col2, col3 = st.columns([8,2,1])
with col2:
    st.write(f'Last refresh: {now_string}')
with col3:
    restart = st.button('Refresh the page')
    if restart:
        st.experimental_rerun()   

st.write('\n')
st.write('\n')
st.write('\n')

col1, col2= st.columns([6,2])
with col1:
    st.title('Churn Customer Prediction Dashboard')
with col2:
    input = st.file_uploader('Upload the new data here:', type='csv')

# Create prediction
if input is not None:
    # Load data as df
    input_df = pd.read_csv(input)

    # Create prediction and encoding
    model_feed = input_df.drop(['customerID', 'TotalCharges'], axis=1)
    encoder = LabelEncoder().fit(current_df['Churn'])
    model = joblib.load('tuned_model_rf.pkl')
    
    churn_data = model.predict(model_feed)
    churn_data = encoder.inverse_transform(churn_data)
    churn_df = pd.DataFrame(churn_data, columns=['Churn'])
    
    # Merge the data
    predicted_df = pd.concat([input_df, churn_df], axis=1)
else:
    predicted_df = pd.DataFrame(columns=current_df.columns)

# Strealit section 2
## Create variables for metrics
current_churn = current_df.loc[current_df['Churn']=='Yes'].shape[0]
predict_churn = predicted_df.loc[predicted_df['Churn']=='Yes'].shape[0]

## Create metrics
col1, col2, col3, col4, = st.columns([1,2,2,1])
st.write('\n')
with col2:
    st.metric('Current churn customers', current_churn)
with col3:
    st.metric('Newly churn customers (predicted)', predict_churn)

## Give an option to choose whether to include predicted data or not for visualization
include_data = st.radio('Include newly inputted data into visualization?', ['yes', 'no'], horizontal=True)

## Create a df based on the option
if include_data == 'yes':
    visualization_df = pd.concat([current_df, predicted_df], axis=0).dropna()
else:
    visualization_df = current_df.dropna()

churn_visualization = visualization_df.loc[visualization_df['Churn']=='Yes']
noChurn_visualization = visualization_df.loc[visualization_df['Churn']=='No']

# Streamlit section 3
st.header('Churn customer distribution by gender and age')
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.plotly_chart(hist_plot_facet(visualization_df, 'Churn', 'SeniorCitizen', 'gender', 'Churn Customer Distribution'))

# Streamlit section 4
st.header('Type of Service')
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
col1, col2 = st.columns([2,1])
with col1:
    selected_columns = st.multiselect('Select type of service to show', service_columns)
with col2:
    viz_type = st.selectbox('Select visualization type:', ['pie', 'bar'])

if viz_type == 'bar':
    for i, cols in enumerate(selected_columns):
        if i%2 == 0:
            cmap = '#2eaa5f'
            col1, col2 = st.columns([1,1])
            with col1:
                st.plotly_chart(hist_plot(churn_visualization, cols, f'{cols} Service for Churn Customer', cmap))
            with col2:
                st.plotly_chart(hist_plot(noChurn_visualization, cols, f'{cols} Service for No-Churn Customer', cmap))
        else:
            cmap = '#d43e1c'
            col1, col2 = st.columns([1,1])
            with col1:
                st.plotly_chart(hist_plot(churn_visualization, cols, f'{cols} Service for Churn Customer', cmap))
            with col2:
                st.plotly_chart(hist_plot(noChurn_visualization, cols, f'{cols} Service for No-Churn Customer', cmap))       

else:
    for i in selected_columns:
        col1, col2 = st.columns([1,1])
        with col1:
            st.plotly_chart(pie_plot(counts_df(churn_visualization, i), 'value', 'name', f'{i} Service for Churn Customer'))
        with col2:
            st.plotly_chart(pie_plot(counts_df(noChurn_visualization, i), 'value', 'name', f'{i} Service for No-Churn Customer'))

# Streamlit section 5
st.header('Churn Distribution by MonthlyCharges and Tenure')
col1, col2 = st.columns([1,1])
with col1:
    st.plotly_chart(hist_plot(churn_visualization, 'MonthlyCharges', 'MonthlyCharges Distibrution for Churn Customer'))
with col2:
    st.plotly_chart(hist_plot(noChurn_visualization, 'MonthlyCharges', 'MonthlyCharges Distibrution for No-Churn Customer'))

col1, col2 = st.columns([1,1])
with col1:
    st.plotly_chart(hist_plot(churn_visualization, 'tenure', 'Tenure Distibrution for Churn Customer'))
with col2:
    st.plotly_chart(hist_plot(noChurn_visualization, 'tenure', 'Tenure Distibrution for No-Churn Customer'))

# Streamlit section 6
## Section to show table
table_view = st.selectbox('Select table view:', ['Detail', 'Compact'])
if table_view =='Compact':
    current_df = current_df[['customerID', 'Churn']]
    predicted_df = predicted_df[['customerID', 'Churn']]

col1, col2 = st.columns([1,1])
with col1:
    st.header('Current data:')
    st.dataframe(current_df)
with col2:
    st.header('Newly predicted data:')
    st.dataframe(predicted_df)