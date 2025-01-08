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

from app1_LOAD import upload_file ,filter_data ,filter_page
from app1_EDA import LOAD_summary_statistics ,plot_frequency_range ,plot_donut_chart ,display_performance_analysis ,display_performance_analysis_filtered
from app1_EDA import filter_negative_counts ,filter_zero_counts ,filter_outliers ,filter_missing_values ,plot_outliers_and_imputed_values ,plot_missing_dates_and_comparisons ,plot_time_series_decomposition
from app1_PREPROCESS import LOAD_impute_dates_and_values ,LOAD_impute_outliers ,preprocessed_data
from app1_MODEL import load_Result, load_Best_Model, plot_actual_vs_predicted ,find_best_model 
from app1_llm import generate_insights_and_recommendations
from app1_PERFORMANCE import load_Result_df ,calculate_confidence_score,plot_results, process_best_model_and_plot




# Page configuration
st.set_page_config(page_title="EU Dashboard",page_icon="📌",layout="wide",initial_sidebar_state="expanded")
alt.themes.enable("dark")


# Load data
def load_data(path):
    df = upload_file(path)
    return df
df = load_data('C:/Users/Shafna/Streamlit/Updated_date_EU.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df['DATE'] = df['DATE'].dt.strftime('%Y-%m-%d')


summ_stat = LOAD_summary_statistics('C:/Users/Shafna/Streamlit/Summary_statistics_EU.csv')
Date_imp = LOAD_impute_dates_and_values('C:/Users/Shafna/Streamlit/Imputed_Dates_EU.csv')
Outl_imp = LOAD_impute_outliers('C:/Users/Shafna/Streamlit/Imputed_Dates&Outliers_EU.csv')
Result_df = load_Result('C:/Users/Shafna/Streamlit/Result_8223.csv')
#Metrcs_df = load_data('C:/Users/Shafna/Streamlit/all_metrics_df.csv')
Best_Model_df = load_Best_Model('C:/Users/Shafna/Streamlit/Best_Models_Metric.csv')

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

df1 = filter_data(df, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1['DATE'] = df1['DATE'].dt.strftime('%Y-%m-%d')
summ_stat1 = filter_data(summ_stat, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
Date_imp1 = filter_data(Date_imp, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
Date_imp1['DATE'] = pd.to_datetime(Date_imp1['DATE'])
Date_imp1['DATE'] = Date_imp1['DATE'].dt.strftime('%Y-%m-%d')
Outl_imp1 = filter_data(Outl_imp, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
Outl_imp1['DATE'] = pd.to_datetime(Outl_imp1['DATE'])
Outl_imp1['DATE'] = Outl_imp1['DATE'].dt.strftime('%Y-%m-%d') 
Result_df1 = filter_data(Result_df, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)
Result_df1['DATE'] = pd.to_datetime(Result_df1['DATE'])
Result_df1['DATE'] = Result_df1['DATE'].dt.strftime('%Y-%m-%d')
Best_Model_df1 = filter_data(Best_Model_df, selected_MARKET, selected_ACCOUNT_ID, selected_CHANNEL_ID, selected_MPG_ID)


df2 = filter_page(df1, selected_MARKET)

data = df1.copy()
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)
data = data.sort_index()


# Check if the data contains at least 24 months
num_months = filtered_data['DATE'].nunique()  
if filtered_data.empty or num_months < 24:
    st.error("Forecasting is not possible due to insufficient data (less than 24 months).")
# Check for empty DataFrames and display messages
elif df1.empty:
    st.error("Forecasting is not possible due to insufficient data ")
elif Result_df1.empty:
    st.error("Forecasting is not possible due to insufficient data.")

else:
    tab = st.tabs(["Leader Board" , "EDA", "Forecast"])

    with tab[0]:
            display_performance_analysis(df)
            
            df_best_model = Best_Model_df1.copy()
            model_counts = df_best_model['MODEL'].value_counts()
            best_model = model_counts.idxmax()
            best_model_count = model_counts.max()
            styled_box = f"""
                <div style="
                    padding: 20px;
                    margin: 20px;
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                    font-family: Arial, sans-serif;
                    text-align: center;
                ">
                    <p style="
                        font-size: 24px; 
                        font-weight: bold; 
                        color: #000000; 
                        margin: 0;
                    ">
                        Overall Best Performing Model - <span style="color: #4CAF50;">{best_model}</span>
                    </p>
                </div>
            """
            st.markdown(styled_box, unsafe_allow_html=True)


    with tab[1]:
        EXAMPLE_NO = 1
        def streamlit_menu(example=3):
            if example == 1:
                with tab[1]:
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
                col = st.columns((10, 2), gap='medium')
                #Downward columns also
                with col[0]:
                    data_1 = df1.copy()
                    data_1['DATE'] = pd.to_datetime(data_1['DATE'])
                    data_1.set_index('DATE', inplace=True)
                    data_1.sort_index(inplace=True)
                    unique_years = data_1.index.year.unique()
                    selected_year = st.session_state.get('selected_year', 2021)  # Default to 2021
                    selected_year = st.selectbox("Select Year", options=unique_years, index=list(unique_years).index(selected_year))
                    filtered_data = data_1[data_1.index.year == selected_year]
                    monthly_sales = filtered_data['AMOUNT'].resample('M').sum()
                    all_months = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-31', freq='M')
                    monthly_sales = monthly_sales.reindex(all_months, fill_value=None)
                    monthly_sales = monthly_sales.dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_sales.index,
                        y=monthly_sales,
                        mode='lines+markers+text',
                        name='Monthly Sales',
                        hoverinfo='x+y+text',
                        marker=dict(size=10)))
                    for date, amount in zip(monthly_sales.index, monthly_sales):
                        fig.add_annotation(
                            x=date,
                            y=amount,
                            text=f'{amount}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40)
                    fig.update_layout(
                        title='Monthly Revenue',
                        xaxis_title='Month',
                        yaxis_title='Amount',
                        xaxis=dict(
                            tickmode='array',
                            tickvals=all_months,
                            tickformat='%b',
                            tickangle=-45
                        ),
                        yaxis=dict(title='Amount'))
                    st.plotly_chart(fig)

                with col[1]:
                    st.markdown('')               
                    with st.expander('About', expanded=True):
                        st.write('''
                            - :blue[**MONTH**]: Period Date
                            - :orange[**AMOUNT**] Revenue Generated From Monthly Sales Transaction 
                            ''')
    
        if selected == "Summary":
                st.title(f" {selected}")
                summary_stats = summ_stat1
                df1 = df1.copy()

                col = st.columns((5,5), gap='medium')
                with col[0]:                  
                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.info(f"  - Zero Count: {row['Zero_Count']}")
                    with st.popover("Zero values"):
                         filter_zero_counts(df1, summary_stats) 
                        

                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        outlier_count = row['Outlier_Count']
                        st.info(f"  - Outlier Count: {outlier_count}")
                        with st.popover("Outlier values"):
                             #filter_outliers(df1, summary_stats)
                             plot_outliers_and_imputed_values(df1, summary_stats, Outl_imp1)
                             st.write("Dataframe Imputed:")
                             filter_outliers(Outl_imp1, summary_stats)
                          

                with col[1]:                    
                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        st.info(f"  - Negative Count: {row['Negative_Count']}")

                    with st.popover("Neagative values"):
                         filter_negative_counts(df1, summary_stats)                     
                


                    st.subheader("")
                    for index, row in summary_stats.iterrows():
                        missing_count = row['Missing_Count']
                        st.info(f"  - Missing Count: {missing_count}")
                        #
                    with st.popover("Missing Dates "):
                        if missing_count >0 :
                            st.write(f"  - Missing Dates: {row['Missing_Values']}")
                            plot_missing_dates_and_comparisons(df1, Date_imp1)
                            #filter_missing_values(Date_imp1, summary_stats)
                        else:
                            st.write("Sorry!....No Missing Dates Present In the Data")
                         
                            
                col = st.columns((50,1), gap='medium')
                with col[0]:                     
                        plot_time_series_decomposition(df1, date_col='DATE', value_col='AMOUNT')
             


        if selected == "Insights":
                st.title(f" {selected}")
                st.write('#### ____________________________________________')
                #display_performance_analysis_filtered(df1)

                # Generate insights and recommendations
                with st.spinner('Generating insights and recommendations LLM Generated...'):
                    all_results_df = Result_df1
                    insights = generate_insights_and_recommendations(df1)
                    st.write(insights)
                    # data_summary, data_result = generate_insights_and_recommendations(df1,all_results_df)              
                    # st.markdown("####  Data summary")
                    # st.write(data_summary)
                    # st.markdown('#### Prediction summary ') 
                    # st.write(data_result)



    with tab[2]:
            st.write('')            
            st.info("Forecast..........")
            all_results_df = Result_df1  
            hitory_df = Outl_imp1 
            # best_model_df = Best_Model_df1
            fig = plot_actual_vs_predicted(hitory_df ,all_results_df)
            st.plotly_chart(fig)
            fig2 = process_best_model_and_plot(hitory_df, all_results_df)
            st.plotly_chart(fig2)



        
        

