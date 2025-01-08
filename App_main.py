import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit_option_menu import option_menu
import time 
import plotly.graph_objects as go 

from LOAD import upload_file ,filter_data ,filter_page
from EDA import calculate_summary_statistics ,plot_frequency_range ,plot_donut_chart ,display_performance_analysis
from PREPROCESS import impute_dates_and_values ,impute_outliers ,preprocessed_data
from MODEL import pred_xgboost_model, pred_random_forest_model, pred_tft_model, plot_actual_vs_predicted
from MODEL import forecast_xgb ,forecast_rf ,forecast_tft ,plot_Forecasted ,plot_Results
from FORECAST import generate_insights_and_recommendations
from PERFORMANCE import calculate_confidence_score, plot_results




# Page configuration
st.set_page_config(page_title="EU Dashboard",page_icon="📌",layout="wide",initial_sidebar_state="expanded")
alt.themes.enable("dark")


# Load data
def load_data(path):
    df = upload_file(path)
    return df
file = 'C:/Users/Shafna/Streamlit/Updated_date_EU.csv'
df = load_data(file)

def load_main_data(df):
    loaded_data = df
    return loaded_data


# Sidebar for filtering
st.sidebar.title('Selection Panel')

MARKET_list = list(df['MARKET'].unique())[::-1]
default_market_id = 8223  
selected_MARKET = st.sidebar.selectbox('MARKET', MARKET_list, index=MARKET_list.index(default_market_id))
df_selected_MARKET = df[df['MARKET'] == selected_MARKET]


# Initialize filtered data
filtered_data = df_selected_MARKET

# Function to check if a group has at least 24 months of data
def has_24_months(group):
    group['DATE'] = pd.to_datetime(group['DATE'])
    num_months = group['DATE'].nunique() 
    return num_months >= 24

# Apply 24-month filter
apply_filter = st.sidebar.radio("Apply 24-Month Filter", ["Yes", "No"], format_func=lambda x: "✅ " + x if x == "Yes" else "❌ " + x)



if apply_filter == 'Yes':
    filtered_data = filtered_data.groupby('ACCOUNT_ID').filter(has_24_months)
    filtered_data = filtered_data.groupby('CHANNEL_ID').filter(has_24_months)
    filtered_data = filtered_data.groupby('MPG_ID').filter(has_24_months)

# Helper function to apply the 24-month filter and update the filtered data
def apply_24_month_filter(data):
    data = data.groupby('ACCOUNT_ID').filter(has_24_months)
    data = data.groupby('CHANNEL_ID').filter(has_24_months)
    data = data.groupby('MPG_ID').filter(has_24_months)
    return data

# Apply filter if selected
if apply_filter == 'Yes':
    filtered_data = apply_24_month_filter(filtered_data)

# Further filtering based on optional selections
ACCOUNT_ID_list = list(filtered_data['ACCOUNT_ID'].unique())[::-1]
selected_ACCOUNT_ID = st.sidebar.selectbox('ACCOUNT_ID', ACCOUNT_ID_list)
if selected_ACCOUNT_ID:
        filtered_data = filtered_data[filtered_data['ACCOUNT_ID']==(selected_ACCOUNT_ID)]
        if apply_filter == 'Yes':
            filtered_data = apply_24_month_filter(filtered_data)


CHANNEL_ID_list = list(filtered_data['CHANNEL_ID'].unique())[::-1]
selected_CHANNEL_ID = st.sidebar.selectbox('CHANNEL_ID', CHANNEL_ID_list)
if selected_CHANNEL_ID:
        filtered_data = filtered_data[filtered_data['CHANNEL_ID']==(selected_CHANNEL_ID)]
        if apply_filter == 'Yes':
            filtered_data = apply_24_month_filter(filtered_data)


MPG_ID_list = list(filtered_data['MPG_ID'].unique())[::-1]
selected_MPG_ID = st.sidebar.multiselect('MPG_ID', MPG_ID_list)
if selected_MPG_ID:
        filtered_data = filtered_data[filtered_data['MPG_ID'].isin(selected_MPG_ID)]
        if apply_filter == 'Yes':
            filtered_data = apply_24_month_filter(filtered_data)


MODEL_list = ["XGBOOST", "RANDOM FOREST REGRESSOR", "TEMPORAL FUSION TRANSFORMER"]
#selected_MODELS = st.multiselect('Select MODELS', MODEL_list, default=MODEL_list)


df1 = filter_data(df, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1['DATE'] = df1['DATE'].dt.strftime('%Y-%m-%d')

df2 = filter_page(df1, selected_MARKET)
#df3 = impute_DATE_page(df2, print_groups=False ,plot_groups=False)
#df = preprocess_page(df2)

# Display the filtered data
# Process filtered data
data = df1.copy()
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)
data = data.sort_index()


# Check if the data contains at least 24 months
num_months = filtered_data['DATE'].nunique()  
if filtered_data.empty or num_months < 24:
    st.error("Forecasting is not possible due to insufficient data (less than 24 months).")


else:
    tab = st.tabs(["EDA", "Forecast", "Performance"])


    with tab[0]:
        EXAMPLE_NO = 1
        def streamlit_menu(example=1):
            if example == 1:
                with tab[0]:
                # with st.sidebar:
                    selected = option_menu(
                        menu_title="Exploratory Data Analysis",  # required
                        options=["History", "Summary", "Insights"],  # required
                        icons=["archive", "clipboard", "lightbulb"],  # optional
                        menu_icon="search",  # optional
                        default_index=0,  # optional
                    )
                return selected

            if example == 2:
                # 2. horizontal menu w/o custom style
                selected = option_menu(
                    menu_title=None,  # required
                    options=["History", "Summary", "Insights"],  # required
                    icons=["archive", "clipboard", "lightbulb"],  # optional
                    menu_icon="search",  # optional
                    default_index=0,  # optional
                    orientation="vertical",
                )
                return selected

            if example == 3:
                # 2. horizontal menu with custom style
                selected = option_menu(
                    menu_title=None,  # required
                    options=["History", "Summary", "Insights"],  # required
                    icons=["archive", "clipboard", "lightbulb"],  # optional
                    menu_icon="search",  # optional
                    default_index=0,  # optional
                    orientation="vertical",
                    styles={
                        "container": {"padding": "0!important", "background-color": "#fafafa"},
                        "icon": {"color": "orange", "font-size": "25px"},
                        "nav-link": {
                            "font-size": "25px",
                            "text-align": "left",
                            "margin": "0px",
                            "--hover-color": "#eee",
                        },
                        "nav-link-selected": {"background-color": "green"},
                    },
                )
                return selected


        selected = streamlit_menu(example=EXAMPLE_NO)

        if selected == "History":
                st.title(f" {selected} ")
                st.write("Filtered Data")
                st.dataframe(df1)


                col = st.columns((2, 4.5, 2), gap='medium')
                #Downward columns also
                with col[0]:
                    st.markdown('#### Rows')
                    no_rows = data.shape[0]
                    st.write(no_rows)

                    st.markdown('#### Total Months')
                    total_months = data.index.nunique()
                    st.write(f"\n\n{total_months}") 

                with col[1]:
                    st.markdown('#### Columns')
                    no_cols = data.shape[1]
                    st.write(no_cols)

                with col[2]:
                    st.markdown('')               
                    with st.expander('About', expanded=True):
                        st.write('''
                            - Data: [EU Dataset](“”)
                            - :orange[**AMOUNT**] Revenue Generated From Monthly Sales Transaction 
                            ''')

                col = st.columns((2, 7), gap='medium')
                with col[0]:
                    st.markdown('#### Features Counts')
                    unique_counts = df.nunique()
                    st.write(unique_counts)
                with col[1]:
                    st.markdown('#### Unique Features')
                    with st.expander("Select Features"):
                        features = list(data.columns)
                        selected_features = st.multiselect('Select features to view unique values', features)
                    if selected_features:
                        for feature in selected_features:
                            st.write(f"Unique values in {feature}")
                            unique_values = data[feature].unique()
                            st.write(unique_values)
                    else:
                        st.write("Please select at least one feature to view unique values.")

        if selected == "Summary":
                st.title(f" {selected}")
                summary_stats = calculate_summary_statistics(data)
                col = st.columns((3, 5, 2), gap='large')
                #Downward columns also
                with col[0]:
                    st.markdown('#### DATE')
                    plot_frequency_range(summary_stats)

                with col[1]:
                    st.markdown('#### FEATURES')
                    #features = ['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']
                    #selected_feature = st.selectbox("Select a Feature for Visualization", features)
                    donut_chart = plot_donut_chart(summary_stats, 'MPG_ID')
                    st.plotly_chart(donut_chart)

                with col[2]:
                    st.markdown('')              
                    with st.expander('About', expanded=True):
                        st.write('''
                            - Data: [EU Dataset](“”)
                            - :orange[**AMOUNT**] Revenue Generated From Monthly Sales Transaction 
                            ''')
                col = st.columns((5,5), gap='medium')
                with col[0]:                  
                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.write(f"**Combination:** {row['MARKET']},{row['ACCOUNT_ID']},{row['CHANNEL_ID']},{row['MPG_ID']}")
                        st.info(f"  - Zero Count: {row['Zero_Count']}") 

                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.write(f"**Combination:** {row['MARKET']},{row['ACCOUNT_ID']},{row['CHANNEL_ID']},{row['MPG_ID']}")
                        outlier_count = row['Outlier_Count']
                        if outlier_count > 0:
                            st.info(f"  - Outlier Count: {outlier_count}")
                            st.write(f"  - Outlier Dates: {row['Outlier_Dates']}")
                        else:
                            st.info(f"  - Outlier Count: {outlier_count}")
                            st.write("  - Outlier Dates: No Outlier Dates")

                with col[1]:                    
                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.write(f"**Combination:** {row['MARKET']},{row['ACCOUNT_ID']},{row['CHANNEL_ID']},{row['MPG_ID']}")
                        st.info(f"  - -ve Count: {row['Negative_Count']}")
                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.write(f"**Combination:** {row['MARKET']},{row['ACCOUNT_ID']},{row['CHANNEL_ID']},{row['MPG_ID']}")
                        missing_count = row['Missing_Count']
                        if missing_count > 0:
                            st.info(f"  - Missing Count: {missing_count}")
                            st.write(f"  - Missing Dates: {row['Missing_Values']}")
                        else:
                            st.info(f"  - Missing Count: {missing_count}")
                            st.write("  - Missing Dates: No Missing Dates")
                            
                col = st.columns((50,1), gap='medium')
                with col[0]:
                    st.markdown('#### Seasonal Decmposition')
                    grouped = data.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])
                    
                    for name, group in grouped:
                        data = group.copy()
                        st.markdown(f"""
                            <div style="background-color: #add8e6; padding: 10px; border-radius: 5px;">
                                <strong>Combination: {name}</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        decomposition = seasonal_decompose(data['AMOUNT'], model='additive', period=1)
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write("Trend Plot:")
                            st.line_chart(decomposition.trend)

                        with col2:
                            st.write("Seasonal Plot:")
                            st.line_chart(decomposition.seasonal)

                        with col3:
                            st.write("Residuals Plot:")
                            st.line_chart(decomposition.resid)
                    # st.markdown('#### PerFormance Summary')
                    # display_performance_analysis(df1)


        if selected == "Insights":
                st.title(f" {selected}")
                st.write('#### ____________________________________________')
                display_performance_analysis(df1)

                if st.button("Insights and Recommendations - LLM Generated"):
                    # Generate insights and recommendations
                    final = preprocessed_data(df, df1)           
                    data_main = load_main_data(final)
                    data, X_test, predictions_xgb, xgb_Forecast,xgb_results_df  = forecast_xgb(data_main)
                            
                    data_main = load_main_data(final)                        
                    data ,X_test ,predictions_rf ,X_test ,predictions_rf ,rf_Forecast ,rf_results_df = forecast_rf(data_main)
                            
                    data_main = load_main_data(final)                        
                    data ,y_test ,predictions_tft ,tft_Forecast ,tft_results_df = forecast_tft(data_main)

                    all_results_df = pd.concat([xgb_results_df, rf_results_df,tft_results_df], ignore_index=True)
                    all_results_df = all_results_df.drop(columns=['APP_NAME', 'CURRENCY_ID'])
                    all_results_df['DATE'] = pd.to_datetime(all_results_df['DATE'])
                    all_results_df['DATE'] = all_results_df['DATE'].dt.strftime('%Y-%m-%d')  
                    # Desired column order
                    new_order = ['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE', 'ACTUAL_AMOUNT', 'TYPE', 'MODEL', 'PREDICTED_AMOUNT']
                    all_results_df = all_results_df[new_order]
                    data_summary, data_result = generate_insights_and_recommendations(df1,all_results_df)
                    col = st.columns((9,2), gap='medium')
                    #Downward columns also
                    with col[0]:
                        st.markdown("####  Data summary")
                        st.write(data_summary)
                        st.markdown('#### Prediction summary ') 
                        st.write(data_result)

                    with col[1]:             
                        with st.expander('About', expanded=True):
                            st.write('''
                                - Data: [EU Dataset](“”)
                                - :orange[**AMOUNT**] Revenue Generated From Monthly Sales Transaction 
                                ''')


    with tab[1]:
            st.write('')
            data_1 = data.copy()
            if not isinstance(data_1.index, pd.DatetimeIndex):
                data_1['DATE'] = pd.to_datetime(data['DATE'])
                data_1.set_index('DATE', inplace=True)
                data_1.sort_index(inplace=True)
            unique_years = data_1.index.year.unique()
            selected_year = st.session_state.get('selected_year', 2021)  # Default to 2021
            selected_year = st.radio("Monthly Revenue", options=unique_years, index=list(unique_years).index(selected_year), horizontal=True)
            filtered_data = data_1[data_1.index.year == selected_year]
            monthly_sales = filtered_data['AMOUNT'].resample('M').sum()
            st.line_chart(monthly_sales)
            # if st.button("Show Preprocessed Data"):
            #     st.write("Filtered df:")
            #     st.write(df1)
            #     st.write("Missing DATE-VALUE Imputed df:")
            #     Prep_df = impute_dates_and_values(df,df1)
            #     Prep_df['DATE'] = pd.to_datetime(Prep_df['DATE'])
            #     Prep_df['DATE'] = Prep_df['DATE'].dt.strftime("%Y-%m-%d")
            #     st.write(Prep_df)
            #     st.write("Outlier Imputated df:")
            #     Prep2_df = impute_outliers(Prep_df)
            #     Prep2_df['DATE'] = pd.to_datetime(Prep2_df['DATE'])
            #     Prep2_df['DATE'] = Prep2_df['DATE'].dt.strftime("%Y-%m-%d")
            #     st.write(Prep2_df)

            st.info("Forecast..........")
            if st.button('Forcast'):
                    start_time = time.time()    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                                
                    progress_text.text(f"Processing models...")
                    time.sleep(1) 
                    final = preprocessed_data(df, df1)           
                    data_main = load_main_data(final)
                    data, X_test, predictions_xgb, xgb_Forecast,xgb_results_df  = forecast_xgb(data_main)
                                    
                    data_main = load_main_data(final)                        
                    data ,X_test ,predictions_rf ,X_test ,predictions_rf ,rf_Forecast ,rf_results_df = forecast_rf(data_main)
                                    
                    data_main = load_main_data(final)                        
                    data ,y_test ,predictions_tft ,tft_Forecast ,tft_results_df = forecast_tft(data_main)

                    all_results_df = pd.concat([xgb_results_df, rf_results_df,tft_results_df], ignore_index=True)
                    all_results_df = all_results_df.drop(columns=['APP_NAME', 'CURRENCY_ID'])
                    all_results_df['DATE'] = pd.to_datetime(all_results_df['DATE'])
                    all_results_df['DATE'] = all_results_df['DATE'].dt.strftime('%Y-%m-%d')  
                    # Desired column order
                    new_order = ['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATE', 'ACTUAL_AMOUNT', 'TYPE', 'MODEL', 'PREDICTED_AMOUNT']
                    all_results_df = all_results_df[new_order]
                    #fig = plot_Forecasted(data, X_test, predictions_xgb, predictions_rf, predictions_tft, xgb_Forecast, rf_Forecast, tft_Forecast)
                    fig = plot_Results(all_results_df)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(all_results_df)

                    start_time = time.time()                
                    for i in range(100):
                            time.sleep(0.1)  
                            progress_bar.progress(i + 1)                   
                    end_time = time.time()
                    processing_time = end_time - start_time           
                    st.success(f"Done processing all models in {processing_time:.2f} seconds!")

            if st.button("Model Predictions"):            
                start_time = time.time()
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                        
                progress_text.text(f"Processing models...")
                time.sleep(1)  # Simulate time delay for demo purposes
                
                final = preprocessed_data(df, df1)
                df = load_main_data(final)
                filtered_grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID'])
                for name, group in filtered_grouped:
                    data = group.copy()
                    data['DATE'] = pd.to_datetime(data['DATE'])
                    data['DATE'] = data['DATE'].dt.strftime("%Y-%m-%d")
                    data1 = data.copy() 


                    X_test ,predictions_xgb = pred_xgboost_model(data1)
                    X_test ,predictions_rf = pred_random_forest_model(data1)
                    y_test ,predictions_tft = pred_tft_model(data1)

                    plot = plot_actual_vs_predicted(X_test, predictions_xgb, predictions_rf, predictions_tft)

                    
                    # st.write(f"y_test shape: {y_test.shape}")
                    # st.write(f"predictions_rf shape: {predictions_rf.shape}")
                    # st.write(f"predictions_xgb shape: {predictions_xgb.shape}")
                    
                    st.write(f"###  PREDICTION ")
                    st.pyplot(plot)
                    
                    # Update progress bar to complete
                    for i in range(100):
                        time.sleep(0.1)  
                        progress_bar.progress(i + 1)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time           
                    st.success(f"Done processing all models in {processing_time:.2f} seconds!")
            # if st.button("Results"):
                            # start_time = time.time()
                
                            # progress_bar = st.progress(0)
                            # progress_text = st.empty()
                                    
                            # progress_text.text(f"Processing models...")
                            # time.sleep(1)  # Simulate time delay for demo purposes                        
                

                            # # Update progress bar to complete
                            # for i in range(100):
                            #     time.sleep(0.1)  
                            #     progress_bar.progress(i + 1)
                            
                            # end_time = time.time()
                            # processing_time = end_time - start_time           
                            # st.success(f"Done processing all models in {processing_time:.2f} seconds!")
                                                


    
        #monthly_sales = data['AMOUNT'].resample('M').sum()
        #st.line_chart(monthly_sales)
        #if st.button('Predict and Evaluate'):
            #results = predict_and_evaluate(data)
            #st.write(results)


    with tab[2]:
        data1 = data.copy()
        st.write('Yearly Revenue')
        monthly_sales = data_1['AMOUNT'].resample('M').sum()
        st.line_chart(monthly_sales)

      
        if st.button("Performance"): 
            final = preprocessed_data(df, df1)
            df = load_main_data(final)
            filtered_grouped = df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'CURRENCY_ID'])
            for name, group in filtered_grouped:
                data = group.copy()
                data['DATE'] = pd.to_datetime(data['DATE'])
                data['DATE'] = data['DATE'].dt.strftime("%Y-%m-%d")
                data1 = data.copy() 


                X_test ,predictions_xgb = pred_xgboost_model(data1)
                X_test ,predictions_rf = pred_random_forest_model(data1)
                y_test ,predictions_tft = pred_tft_model(data1)

                # Simulated time index
                time_index = X_test.index

                mean_predictions, actual, std_dev, mae, confidence_score, lower_bound, upper_bound = calculate_confidence_score(predictions_rf, predictions_xgb, predictions_tft, y_test)
                
                st.info(f"Confidence Score: {confidence_score:.2f}%")
                
                fig = plot_results(mean_predictions, actual, lower_bound, upper_bound, time_index, confidence_score)
                st.plotly_chart(fig, use_container_width=True)

             

        
        

