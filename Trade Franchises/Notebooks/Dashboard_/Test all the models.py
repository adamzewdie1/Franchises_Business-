
# This notebook is to test all the models and display them in interactive dashboard 
# I already did dashboard for the EDA 
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from prophet import Prophet 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import matplotlib.pyplot as plt

#initializing the tabs

tab1,tab2,tab3=st.tabs(["📈Revenue Forcasting", " 📊 Store Segmentation", "⚠️Store Abondnment Prediction"])


with tab1:
    st.header("🏪 Upcoming Biweekly revenue forcast")
    #start with prophet 

    rev=pd.read_csv(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Meta Data\Raw data\biweekly_revenue_data.csv")  
    load_pro=joblib.load(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Notebooks\Models\prophet_model.pkl")

    # need to give options. based on store number 
    #1) need to convert store_number to customer number . make stores to 100 
    #2) users can insert up to 100 stores. 
    #3) give next week data 
 
    rev=rev.drop(rev[rev["store_id"].isin([1,2,3,4,5,6,7,8,9])].index)
    rev=rev.drop(columns="store_id", axis=1)
    rev.rename(columns={"customer_id":"Store_id"}, inplace=True)
    temp=rev
    new_rev=temp[temp["Store_id"]==18]
    new_rev["revenue"]=0
    new_rev=new_rev.rename(columns={"ds":"date","y":"Store_id"})
        
    un=rev["Store_id"].unique()
    
    for n in range(len(un)):
        temp1=rev[rev["Store_id"]==un[n]]
        temp1=temp1[["date","revenue"]]
        ave=temp1["revenue"].mean()
        temp1["revenue"]=temp1["revenue"].replace(0,ave)
        temp1=temp1.rename(columns={"date":"ds","revenue":"y"})
        sa1=Prophet(changepoint_prior_scale=0.5)
        sa1.add_seasonality(name="biweekly", period=28, fourier_order=3)
        sa1.fit(temp1)
        future=sa1.make_future_dataframe(periods=1, freq='14D')
        forcast=sa1.predict(future)
        last=forcast.tail(1)
        time=last.iloc[0,0]
        number=last.iloc[0,2]
        new_rev.iloc[n,0]=time
        new_rev.iloc[n,1]=n
        new_rev.iloc[n,2]=number
        
    
    new_rev["Store_id"]=new_rev["Store_id"].astype('int64')
    fig=px.bar(new_rev,x="Store_id",y="revenue")
    st.plotly_chart(fig)
    new_rev=new_rev[new_rev["Store_id"]<10]
    new_rev


with tab2:
    st.header(" 📊 Store Segmentation")
    rev=pd.read_csv(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Meta Data\Raw data\Revenue.csv")
    

    #inhouse data reareagning the numbers 
    temp_12m=rev["Total_Revenue_12mo"].copy()
    temp_6m=rev["Total_Revenue_6mo"].copy()
    temp_3m=rev["Total_Revenue_3mo"].copy()
    
    for n in range(len(rev["Total_Revenue_12mo"])):
        t12=temp_12m[n]
        t6=temp_6m[n]
        t3=temp_3m[n]
        temp=[t12,t6,t3]
        temp.sort()  
        temp_12=temp[2]
        temp_6=temp[1]
        temp_3=temp[0]
        temp_12m[n]=temp_12
        temp_6m[n]=temp_6
        temp_3m[n]=temp_3
        temp=pd.NA
    
    rev["Total_Revenue_12mo"]=temp_12m
    rev["Total_Revenue_6mo"]=temp_6m
    rev["Total_Revenue_3mo"]=temp_3m

    #scaling the data 
    ints=rev.select_dtypes(include=['int64']).columns.tolist()
    scaler=StandardScaler()
    ints_s=scaler.fit_transform(rev[ints])
    X=pd.DataFrame(ints_s,columns=scaler.get_feature_names_out(ints))

    #kmeans model loaded
    load=joblib.load(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Notebooks\Models\kmeans.pkl")
    ar=load.fit_predict(X)

    #pca to show 
    pca=PCA(n_components=2)
    x_cpa=pca.fit_transform(X)
    figu, ax = plt.subplots(figsize=(8,6))
    scatter=ax.scatter(x_cpa[:,0],x_cpa[:,1],c=ar, cmap="viridis", alpha=0.7)
    ax.set_xlabel=("pca compnent 1")
    ax.set_ylabel=("pca compnent 2")
    ax.set_title=("The model Dividing the Stores into 3 Catagories")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    st.pyplot(figu)

    #creating 
    rev["Rev_Catagory"]=ar
    rev["Rev_Catagory"].replace({1:"High Performer",
                            0:"Low Performer",
                            2:"Consistance Performer"},inplace=True)
    #st.title("")
    rev=rev.drop(columns="Revenue_Trend_Flag", axis=1)
    rev

with tab3:
    st.header("⚠️Store Abondnment Prediction")

    #get the data ready
    X=pd.read_csv(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Meta Data\Raw data\readyx.csv")
    X=X.drop(columns="Unnamed: 0",axis=1)
    
    #load the model. 
    load2=joblib.load(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Notebooks\Models\xgbost_model.pkl")
    lo2=load2.predict_proba(X)[:,1]
    lo3=(lo2>0.1).astype(int)
    X["Abondonment Risk Level"]=lo3
    X["Abondonment Risk Level"].replace({0:"Low Risk",1:"High Risk"},inplace=True)
    X["Store_ID"]=rev["Store_ID"]
    y=X[["Store_ID","Abondonment Risk Level"]]
    y
    pay=pd.read_csv(r"C:\Users\silve\Desktop\Data Science\Projects\Trade Franchises\Meta Data\Raw data\Payments.csv")
    
    st.title("Store Missed Payment Data")
    pay










    