import streamlit as st
dataset_loc="dataset/churn.csv"
img_loc = "img/download.png"
img_loc2 = "img/N.jpg"
st.image(img_loc, use_column_width = True)
st.markdown("<h1 style='text-align: center; color: red;'>NETFLIX Churn Prediction</h1>", unsafe_allow_html=True)
st.sidebar.title('Login')
st.sidebar.image(img_loc2, use_column_width = True)
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input('Password')
st.sidebar.warning("NEtflix hai mamu :sunglasses:")
button_was_clicked = st.sidebar.button('Login')
Gender = st.selectbox("Gender: ",
                     ['Male', 'Female', 'Others'])
SeniorCitizen = st.selectbox("SeniorCitizen: ",
                     [0,1])
if SeniorCitizen==0:
        st.write("You are not a SeniorCitizen")
else:
        st.write("You are a SeniorCitizen")
Partner = st.selectbox("Partner: ",
                     ['Yes','No'])
if Partner=='No':
        st.write("You don't have a Partner")
else:
        st.write("You have a Partner")
Dependents = st.selectbox("Dependents: ",
                     ['Yes','No'])
if Dependents=='No':
        st.write("You don't have a Dependents")
else:
        st.write("You have a Dependents")
Tenure=st.number_input("Enter your Tenure from ", 1, 100)
PhoneService = st.selectbox("PhoneService: ",
                     ['Yes','No'])
if PhoneService=='No':
        st.write("You don't have PhoneService")
else:
        st.write("You have PhoneService")
MultipleLines = st.selectbox("MultipleLines: ",
                     ['Yes','No'])
if MultipleLines=='No':
        st.write("You don't have MultipleLines")
else:
        st.write("You have MultipleLines")
Internetservices = st.selectbox("Internetservices: ",
                     ['DSL', 'Fibre optics', 'No'])
OnlineSecurity = st.selectbox("OnlineSecurity : ",
                     ['Yes','No'])
if OnlineSecurity=='No':
        st.write("You don't have OnlineSecurity")
else:
        st.write("You have OnlineSecurity")
OnlineBackup = st.selectbox("OnlineBackup : ",
                     ['Yes','No'])
if OnlineBackup=='No':
        st.write("You don't have OnlineBackup")
else:
        st.write("You have OnlineBackup")
DeviceProtection = st.selectbox("DeviceProtection : ",
                     ['Yes','No'])
if DeviceProtection=='No':
        st.write("You don't have DeviceProtection")
else:
        st.write("You have DeviceProtection")
Techsupport = st.selectbox("Techsupport : ",
                     ['Yes','No'])
if Techsupport=='No':
        st.write("You don't have a Tech support")
else:
        st.write("You have a Tech support")
StreamingTV = st.selectbox("StreamingTV : ",
                     ['Yes','No'])
if StreamingTV=='No':
        st.write("You are not Streaming in TV")
else:
        st.write("You are Streaming in TV")
StreamingMovies = st.selectbox("StreamingMovies : ",
                     ['Yes','No'])
if StreamingMovies=='No':
        st.write("You don't have a StreamingMovies")
else:
        st.write("You have a StreamingMovies")
ContractType = st.selectbox("ContractType : ",
                     ['Month to month','One year','Two year'])
if ContractType=='One year':
        st.write("You have contract of  one year")
elif ContractType=='Two year':
        st.write("You have contract of two year")
else:
        st.write("You have contract of month to month ")
PaperlessBilling = st.selectbox("PaperlessBilling : ",
                     ['Yes','No'])
if PaperlessBilling=='No':
        st.write("You don't have a PaperlessBilling")
else:
        st.write("You have a Pa")
PaymentMethod = st.selectbox("PaymentMethod : ",
                     ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])

MonthlyCharges=st.number_input("Enter your MonthlyCharges charges", 1.0, 1000.0)
TotalCharges=st.number_input("Enter your TotalCharges charges", 1.0, 10000.0)
import pandas as pd
column_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
df = pd.DataFrame(columns = column_names)
df['gender']= Gender
df['SeniorCitizen']=SeniorCitizen
df['Partner']=Partner
df['Dependents']=Dependents
df['Tenure']= Tenure
df['PhoneService']=PhoneService
df['MultipleLines']=MultipleLines
df['InternetService']=Internetservices
df['OnlineSecurity']=OnlineSecurity
df['OnlineBackup']=OnlineBackup
df['DeviceProtection']=DeviceProtection
df['TechSupport']=Techsupport
df['StreamingTV']=StreamingTV
df['StreamingMovies']=StreamingMovies
df['Contract']=ContractType
df['PaperlessBilling']=PaperlessBilling
df['PaymentMethod']=PaymentMethod
df['MonthlyCharges']=MonthlyCharges
df['TotalCharges']=TotalCharges
def clean_df(df):
    df.loc[:, ['Partner', 'Dependents', 'PhoneService', \
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', \
            'TechSupport', 'StreamingTV', 'StreamingMovies',\
            'PaperlessBilling']]
    var = ['Partner', 'Dependents', 'PhoneService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling']
    for feature in var:
        df[feature] = df[feature].apply(lambda x : 1 if x=='Yes' else 0)
    df['TotalCharges'] = df['TotalCharges'].astype('float64')
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df), columns=encoder.get_feature_names(df.columns))
    df = pd.concat([df, df_encoded], axis=1)
    df = df.drop('gender', 'MultipleLines', 'InternetService', 'Contract','PaymentMethod', axis=1)

def load_data(dataset_loc):
    df = pd.read_csv(dataset_loc)
    df.drop('Churn', axis=1, inplace=True)
    return df

def load_description(df):
    st.header("Data Preview")
    preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
    if(preview == "Top"):
        st.write(df.head())
    if(preview == "Bottom"):
        st.write(df.tail())

    #display the whole dataset
    if(st.checkbox("Show complete Dataset")):
         st.write(df)
    # Show shape
    if(st.checkbox("Display the shape")):
        st.write(df.shape)
        dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
        if(dim == "Rows"):
            st.write("Number of Rows", df.shape[0])
        if(dim == "Columns"):
            st.write("Number of Columns", df.shape[1])

    # show columns
    if(st.checkbox("Show the Columns")):
        st.write(df.columns)


def standardization (df):
    from sklearn.preprocessing import StandardScaler
    standardized_data = StandardScaler().fit_transform(df)
    df2 = pd.DataFrame(standardized_data,columns = features)
def predict(df2):
    from pickle import dump,load
    classifier=load(open('pickle/dump.pkl','rb'))
    prediction = classifier.predict(df2)
click = st.button('SUBMIT')
if click:
    def main():
        page_bg_img ='''
            <style>
            body {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
            }
            </style>
            '''

        st.markdown(page_bg_img, unsafe_allow_html=True)
        df = load_data(dataset_loc)
        load_description(df)
        clean_df(df)
        standardization(df)
        predict(df2)
    if predict==0:
        st.write('Churned')
    elif predict==1:
        st.write('Not Churned')
    else:
        st.write('None')
# if(__name__=='__main__'):
    # main()
