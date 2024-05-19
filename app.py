
""" MIT License Copyright (C) 2024 Miroslaw Steblik """

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl

from investor_reporting_app.calculations import MonthlyReturnSeries, DailyPriceSeries
from investor_reporting_app.utils import is_eom, validate_date_range, validate_reporting_date, validate_date_existance



#------------------------------- PAGE CONFIG -------------------------------------------------------#
def style_table(df, precision, width=1000):
    return (df.style.format("{:." + str(precision) + "%}")
            .set_properties(**{'color': 'white'})), width

st.set_page_config(page_title= "Investor Reporting App", layout="wide")

#----------------------------------- IMAGES --------------------------------------------------------#
image_list = ['resources/bar_chart_1.png',
              'resources/line_chart_2.png',
              'resources/bar_chart_3.png',
              'resources/bar_chart_2.png',
              'resources/histogram.png']

#----------------------------------- CSV INPUT -----------------------------------------------------#
st.sidebar.title('Menu')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file to analyze", type="csv")
st.sidebar.markdown("")

if uploaded_file is None:
    st.sidebar.info('Please upload a CSV file to start the analysis.')

#----------------------------------- INTRO ---------------------------------------------------------#
    st.header('Investor Reporting App')
    st.subheader('Simplify reporting workflow and reduce your manual efforts with interactive dashboard.')

    # Display the images
    cols = st.columns([1,1,1,1,1])
    for col, image in zip(cols, image_list):
        col.image(image)  

    st.markdown('#### Inside the app you can find the following functionalities:')

    df = pd.DataFrame({
        'Tables': ['Cumulative and annualized performance tables', 'Volatility tables', 'Calendar performance table'],   
        'Charts': ['Price line chart', 'Cumulative performance chart', 'Other']
        })
    st.data_editor(df, height=150, width=700, hide_index=True)


    with st.popover("More Information"):
        st.markdown("""
            Investment team can upload a CSV file containing the daily price series or monthly returns of the instruments and choose the desired option from the sidebar menu. 
            
            App will transform data loaded from the file into ready to read format that can be used for on-the-fly analysis or as a part of client reporting.

            ## *:blue[No more excel sheets, no more manual calculations]*
            """)


    st.write('---')



    st.stop()

#----------------------------------- CHOICE ------------------------------------#
options = ['Monthly Returns', 'Daily Price Series']
choice = st.sidebar.radio("Choose option", options, label_visibility="collapsed")
st.sidebar.markdown("") # Add space

#----------------------------------- SLIDER ------------------------------------#
precision = st.sidebar.slider('Select the number of decimal places', 0, 5, 2)

#----------------------------------- DATE INPUT --------------------------------#
st.sidebar.markdown("") # Add space
reporting_date = st.sidebar.date_input('Reporting Date', pd.to_datetime('31/01/2024', format='%d/%m/%Y')) #remove hardcoding
since_inception_date = st.sidebar.date_input('Since Inception Date', pd.to_datetime('21/11/2014', format='%d/%m/%Y'))

#----------------------------------- CONTAINERS ----------------------------------------------------#
sub_title1_cont = st.container()
sub_title2_cont = st.container()

row1 = st.columns(2)
row2 = st.columns(2)

first_container = row1[0].container(border=True)
first_container.text("Cumulative Performance")

second_container = row1[1].container(border=True)
second_container.text("Annualized Performance")

third_container = row2[0].container(border=True)
third_container.text("Annualized Volatility")

fourth_container = row2[1].container(border=True)
fourth_container.text("Calendar Performance")

if choice == 'Daily Price Series':
    fifth_container = st.container(border=True)
    fifth_container.text("Price Chart")

    sixth_container = st.container(border=True)
    sixth_container.text("Histogram")

#---------------------------------- FIRST CONTAINER ----------------------------#
def create_cumulative_performance_container(performance):
    cumulative_performance = performance.calculate_cumulative_performance().copy()
    cumulative_performance_display = cumulative_performance.reset_index()
    cumulative_performance_display = cumulative_performance_display.rename(columns={'index': 'Performance Metric', 'Performance': 'Cumulative Performance'})
    cumulative_performance_melted = cumulative_performance_display.melt(id_vars='Performance Metric', var_name='Instrument', value_name='Cumulative Performance')
    fig = px.bar(cumulative_performance_melted, x='Instrument', y='Cumulative Performance', 
                    color='Performance Metric', barmode='group', title='Cumulative Performance')
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="")
    fig.update_xaxes(title_text="")
    return fig, cumulative_performance

#---------------------------- SECOND CONTAINER ---------------------------------#
def create_annualized_performance_container(performance):
    annualized_performance = performance.calculate_annualized_performance().copy()
    annualized_performance_display = annualized_performance.reset_index()
    annualized_performance_display = annualized_performance_display.rename(columns={'index': 'Performance Metric', 'Performance': 'Annualized Performance'})
    annualized_performance_melted = annualized_performance_display.melt(id_vars='Performance Metric', var_name='Instrument', value_name='Annualized Performance')
    fig = px.bar(annualized_performance_melted, x='Instrument', y='Annualized Performance', 
                    color='Performance Metric', barmode='group', title='Annualized Performance')
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="")
    fig.update_xaxes(title_text="")
    return fig, annualized_performance

#------------------------------- THIRD CONTAINER -------------------------------#
def create_annualied_volatility_container(performance):
    color_discrete_map = {'1 year': 'lightblue', '3 year': 'skyblue', '5 year': 'deepskyblue', '10 year': 'dodgerblue', 'Since Inception': 'darkblue'}
    
    annualized_volatility = performance.calculate_annualized_volatility().copy()
    annualized_volatility_display = annualized_volatility.reset_index()
    annualized_volatility_display = annualized_volatility_display.rename(columns={'index': 'Performance Metric'})
    annualized_volatility_melted = annualized_volatility_display.melt(id_vars='Performance Metric', var_name='Instrument', value_name='Volatility')
    fig = px.bar(annualized_volatility_melted, x='Instrument', y='Volatility', 
                    color='Performance Metric', barmode='group', title='Annualized Volatility',
                    color_discrete_map=color_discrete_map)
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="") 
    fig.update_xaxes(title_text="") 
    
    return fig, annualized_volatility

#-------------------------- FOURTH CONTAINER -----------------------------------#
def create_calendar_performance_container(performance):
    
    calendar_performance = performance.calculate_calendar_performance().copy()
    # Check if 'Date' column exists, if not create it using the DataFrame's index
    if 'Date' not in calendar_performance.columns:
        calendar_performance['Date'] = calendar_performance.index  
    calendar_performance_melted = calendar_performance.melt(id_vars='Date', var_name='Instrument', value_name='Performance')
    
     # Create a color map
    bars = sorted(calendar_performance_melted['Instrument'].unique()) #colors per bar
    colors = cl.scales[str(len(bars))]['qual']['Set1']
    color_map = {bar: color for bar, color in zip(bars, colors)}
    
    fig = px.bar(calendar_performance_melted, x='Date', y='Performance', 
                 color='Instrument', barmode='group', title='Calendar Performance',
                 color_discrete_map=color_map)
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="")
    fig.update_xaxes(title_text="")
    calendar_performance = calendar_performance.drop(columns='Date')
    return fig, calendar_performance

#------------------------- FIFTH CONTAINER -------------------------------------#
def create_price_chart_container(performance):
    data = performance.fund_data.ffill().copy()
    traces = [go.Scatter(x=data.index, y=data[instrument], name=instrument) for instrument in data.columns]
    layout = go.Layout(
        title_text="",
        legend_title_text="",
        height=400,
        width=1400
        )
    fig = go.Figure(data=traces, layout=layout)
    fig.update_yaxes(title_text="")
    return fig

#------------------------- SIXTH CONTAINER -------------------------------------#
def create_histogram_container(performance):
    histogram_columns = st.columns([1,1,1])
    data = performance.fund_data.ffill().copy()
    data = data.pct_change().dropna()
    for col, instrument in zip(histogram_columns, data.columns):
        fig = px.histogram(data[instrument], x=instrument, title=instrument)
        fig.update_layout( height=500,width= 400, title_text="",legend_title_text='')
        fig.update_yaxes(title_text="")
        fig.update_xaxes(tickformat=".1%")
        min_value = data[instrument].min()
        max_value = data[instrument].max()
        fig.add_annotation(x=min_value, y=25, text=f'Min: {min_value:.1%}', showarrow=False)
        fig.add_annotation(x=max_value, y=25, text=f'Max: {max_value:.1%}', showarrow=False)
        col.plotly_chart(fig)
    return fig

#----------------------------------------- FOOTER --------------------------------------------------#
st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub License](https://img.shields.io/github/license/miroslaw-steblik/investor-reporting-app)]"
    "(https://github.com/miroslaw-steblik/investor-reporting-app)"
    )

#----------------------------------------- MAIN ----------------------------------------------------#
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if choice == 'Monthly Returns':  

        #-------------------------- Monthly Return Series-----------------------#
        performance_monthly = MonthlyReturnSeries(
                                        data=data,
                                        reporting_date=reporting_date, 
                                        since_inception_date=since_inception_date)
                                            
        # Validate the dates
        if not is_eom(reporting_date, '%d/%m/%Y'):
            st.error('Please select month end date.')
            st.stop()
        elif validate_date_range(reporting_date, '%d/%m/%Y', data) or validate_date_range(since_inception_date, '%d/%m/%Y', data):
            st.error('Date is out of data range.')
            st.stop()
        elif validate_reporting_date(reporting_date, since_inception_date):
            st.error('Reporting date cannot be earlier than inception date.')
            st.stop()
        elif validate_date_existance(since_inception_date, '%d/%m/%Y', data) or validate_date_existance(reporting_date, '%d/%m/%Y', data):
            st.error('Selected date does not exist. Please choose different date.')
            st.stop()


        with sub_title1_cont:
            st.header(':orange[_Monthly Returns_]')

        with first_container:
            fig, cumulative_performance = create_cumulative_performance_container(performance_monthly)
            st.plotly_chart(fig)
            cumulative_performance = performance_monthly.calculate_cumulative_performance()
            st.dataframe(*style_table(cumulative_performance, precision))

        with second_container:
            fig, annualized_performance = create_annualized_performance_container(performance_monthly)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_performance, precision))

        with third_container:
            fig, annualized_volatility = create_annualied_volatility_container(performance_monthly)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_volatility, precision))

        with fourth_container:
            fig, calendar_performance = create_calendar_performance_container(performance_monthly)
            st.plotly_chart(fig)
            calendar_performance = calendar_performance.head(5)
            calendar_performance.index = calendar_performance.index.astype(int).astype(str)
            st.dataframe(*style_table(calendar_performance, precision))
        
        
    else:
        #-------------------------- Daily Price Series -------------------------#
        performance_daily = DailyPriceSeries(data=data,
                                            reporting_date=reporting_date, 
                                            since_inception_date=since_inception_date)
        
        # Validate the dates
        if not is_eom(reporting_date, '%d/%m/%Y'):
            st.error('Reporting Date: Please select month end date.')
            st.stop()
        elif validate_date_range(reporting_date, '%d/%m/%Y', data) or validate_date_range(since_inception_date, '%d/%m/%Y', data):
            st.error('Date is out of data range.')
            st.stop()
        elif validate_reporting_date(reporting_date, since_inception_date):
            st.error('Reporting date cannot be earlier than inception date.')
            st.stop()
        elif validate_date_existance(since_inception_date, '%d/%m/%Y', data):
            st.error('Since Inception date does not exist. Please choose different date.')
            st.stop()
        

        with sub_title2_cont:
            st.header(':orange[_Daily Price Series_]')

        with first_container:
            fig, data = create_cumulative_performance_container(performance_daily)
            st.plotly_chart(fig)
            cumulative_performance = performance_daily.calculate_cumulative_performance()
            st.dataframe(*style_table(cumulative_performance, precision))

        with second_container:
            fig, annualized_performance = create_annualized_performance_container(performance_daily)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_performance, precision))

        with third_container:
            fig, annualized_volatility = create_annualied_volatility_container(performance_daily)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_volatility, precision))

        with fourth_container:
            fig, calendar_performance = create_calendar_performance_container(performance_daily)
            st.plotly_chart(fig)
            calendar_performance = calendar_performance.head(5)
            calendar_performance.index = calendar_performance.index.astype(int).astype(str)
            st.dataframe(*style_table(calendar_performance, precision))

        if choice == 'Daily Price Series':
            with fifth_container:
                fig = create_price_chart_container(performance_daily)
                st.plotly_chart(fig)
          
            with sixth_container:
                create_histogram_container(performance_daily)







                



            













            
            




