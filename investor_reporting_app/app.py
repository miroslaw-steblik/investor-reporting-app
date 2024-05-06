
""" MIT License, Copyright (C) 2024, Miroslaw Steblik """

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dateutil.relativedelta import relativedelta
from QuantLib import Date, Thirty360 # 30/360 is applied to follow the industry standard for calculating the time period between two dates
import plotly.graph_objects as go


#------------------------------- VALIDATION FUNCTIONS ----------------------------------------------#
    
def validate_reporting_date(reporting_date, since_inception_date):
    if reporting_date < since_inception_date:
        return True             # error occurs

def validate_date_range(date, date_format, data):
    date = pd.to_datetime(date, format=date_format)
    min_date = pd.to_datetime(data.index.min(), unit='s')
    max_date = pd.to_datetime(data.index.max(), unit='s')
    if date < min_date or date > max_date:
        return True           # error occurs
    
def is_eom(date, date_format):
    date = pd.to_datetime(date, format=date_format)
    return date.is_month_end

def validate_date_existance(date, date_format, data):
    date = pd.to_datetime(date, format=date_format)
    if date not in data.index:
        return True


#----------------------------- Monthly Return Series Class -----------------------------------------#

class MonthlyReturnSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = data
        self.date_format = '%d/%m/%Y'
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format )
        self.reporting_date = pd.to_datetime(reporting_date, format=self.date_format )
        self.since_inception_date = pd.to_datetime(since_inception_date, format=self.date_format )
        self.fund_data.set_index('Date', inplace=True)
        
        #mask
        self.one_year_period = self.fund_data.loc[(self.fund_data.index > (self.reporting_date - pd.DateOffset(months=12))) & (self.fund_data.index <= self.reporting_date)] # 1 year period mask
        self.three_year_period = self.fund_data.loc[(self.fund_data.index > (self.reporting_date - pd.DateOffset(months=36))) & (self.fund_data.index <= self.reporting_date)] # 3 years period mask
        self.five_year_period = self.fund_data.loc[(self.fund_data.index > (self.reporting_date - pd.DateOffset(months=60))) & (self.fund_data.index <= self.reporting_date)]# 5 years period mask
        self.ten_year_period = self.fund_data.loc[(self.fund_data.index > (self.reporting_date - pd.DateOffset(months=120))) & (self.fund_data.index <= self.reporting_date)]# 10 years period mask
        self.since_inception_period = self.fund_data.loc[(self.fund_data.index >= self.since_inception_date) & (self.fund_data.index <= self.reporting_date)] # Since inception period mask

        # Calculate the period for the since_inception_performance
        start_date = Date(self.since_inception_date.day, self.since_inception_date.month, self.since_inception_date.year)
        end_date = Date(self.reporting_date.day, self.reporting_date.month, self.reporting_date.year)
        since_inception_period = Thirty360(Thirty360.European).yearFraction(start_date, end_date)
        self.periods = {0: 1, 1: 3, 2: 5, 3: 10, 4: since_inception_period}  # The time periods for each performance metric
        self.period_names = {0: '1 year', 1: '3 year', 2: '5 year', 3: '10 year', 4: 'Since Inception'}


    def calculate_cumulative_performance(self):
        self.fund_data = self.fund_data.resample('ME').last()
        one_year_performance = (self.one_year_period   + 1).prod() - 1
        three_year_performance = (self.three_year_period  + 1).prod() - 1
        five_year_performance = (self.five_year_period  + 1).prod() - 1
        ten_year_performance = (self.ten_year_period  + 1).prod() - 1
        since_inception_performance = (self.since_inception_period  + 1).prod() - 1
        performance = pd.concat([one_year_performance, three_year_performance, five_year_performance, ten_year_performance, since_inception_performance], axis=1).T
        performance = performance.rename(index=self.period_names)
        return performance
        # tested and working


    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}     # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        # tested and working

    def calculate_annualized_volatility(self):
        self.fund_data = self.fund_data.resample('ME').last()
        one_year_volatility = self.one_year_period .std(ddof=0) * np.sqrt(12)
        three_year_volatility = self.three_year_period .std(ddof=0) * np.sqrt(12)
        five_year_volatility = self.five_year_period .std(ddof=0) * np.sqrt(12)
        ten_year_volatility = self.ten_year_period .std(ddof=0) * np.sqrt(12)
        since_inception_volatility = self.since_inception_period .std(ddof=0) * np.sqrt(12)
        volatility = pd.concat([one_year_volatility, three_year_volatility, five_year_volatility, ten_year_volatility, since_inception_volatility], axis=1).T
        volatility = volatility.rename(index=self.period_names)
        return volatility
        # tested and working

    def calculate_calendar_performance(self):
        yearly_data = self.fund_data.resample('YE')   
        performance_list = []
        dates_list = []
        for year, data in yearly_data:
            if len(data) ==12:  # Check if the year has 12 months of data
                calculate_performance = (data + 1).prod() - 1
                performance_list.append(calculate_performance)
                dates_list.append(year.year)
        performance_df = pd.concat(performance_list, axis=1).T  # Transpose the DataFrame
        performance_df.columns = self.fund_data.columns
        performance_df.index = dates_list
        performance_df = performance_df.sort_index(ascending=False).head(5)
        return performance_df
        # tested and working

#----------------------------- Daily Price Series Class --------------------------------------------#
class DailyPriceSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = data
        self.date_format = '%d/%m/%Y'
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format )
        self.reporting_date = pd.to_datetime(reporting_date, format=self.date_format )
        self.since_inception_date = pd.to_datetime(since_inception_date, format=self.date_format )
        self.fund_data.set_index('Date', inplace=True)

        # Calculate the period for the since_inception_performance
        start_date = Date(self.since_inception_date.day, self.since_inception_date.month, self.since_inception_date.year)
        end_date = Date(self.reporting_date.day, self.reporting_date.month, self.reporting_date.year)
        since_inception_period = Thirty360(Thirty360.European).yearFraction(start_date, end_date)
        self.periods = {0: 1, 1: 3, 2: 5, 3: 10, 4: since_inception_period}  # The time periods for each performance metric
        self.period_names = {0: '1 year', 1: '3 year', 2: '5 year', 3: '10 year', 4: 'Since Inception'}


    def calculate_cumulative_performance(self):
         
            daily_fund_data = self.fund_data.asfreq('D').ffill()  # Use daily data
            since_inception_value = daily_fund_data.loc[self.since_inception_date]
            monthly_fund_data = self.fund_data.resample('ME').last()  # Use month-end data
            reporting_date_value = monthly_fund_data.loc[self.reporting_date]

            since_inception_performance = (reporting_date_value / since_inception_value) - 1
            one_year_performance = (reporting_date_value / monthly_fund_data.loc[self.reporting_date - relativedelta(years=1)]) -1
            three_year_performance = (reporting_date_value / monthly_fund_data.loc[self.reporting_date - relativedelta(years=3)]) -1
            five_year_performance = (reporting_date_value / monthly_fund_data.loc[self.reporting_date - relativedelta(years=5)]) -1
            ten_year_performance = (reporting_date_value / monthly_fund_data.loc[self.reporting_date - relativedelta(years=10)]) -1 
            performance = pd.concat([one_year_performance, three_year_performance, five_year_performance, ten_year_performance, since_inception_performance], axis=1).T
            performance = performance.rename(index=self.period_names)
            return performance


    
    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}      # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        # tested and working


    def calculate_annualized_volatility(self): 
        self.fund_data = data
        self.fund_data = self.fund_data.ffill()  # Forward fill any missing values
        returns = self.fund_data.pct_change()  

        one_year_volatility = returns.loc[self.reporting_date - pd.DateOffset(years=1,days=-1):self.reporting_date].std(ddof=0) * np.sqrt(252)
        three_year_volatility = returns.loc[self.reporting_date - pd.DateOffset(years=3, days=-1):self.reporting_date].std(ddof=0) * np.sqrt(252)
        five_year_volatility = returns.loc[self.reporting_date - pd.DateOffset(years=5,days=-1):self.reporting_date].std(ddof=0) * np.sqrt(252)
        ten_year_volatility = returns.loc[self.reporting_date - pd.DateOffset(years=10,days=-1):self.reporting_date].std(ddof=0) * np.sqrt(252)
        since_inception_volatility = returns.loc[self.since_inception_date:self.reporting_date].std(ddof=0) * np.sqrt(252)
        volatility = pd.concat([one_year_volatility, three_year_volatility, five_year_volatility, ten_year_volatility, since_inception_volatility], axis=1).T
        volatility = volatility.rename(index=self.period_names)
        return volatility


    def calculate_calendar_performance(self):
        monthly_data = self.fund_data.resample('ME').last()
        years_with_12_months = monthly_data.groupby(monthly_data.index.year).size() == 12
        valid_years = years_with_12_months[years_with_12_months].index
        valid_data = self.fund_data[self.fund_data.index.year.isin(valid_years)]  # Filter the data by the valid years
        yearly_data = valid_data.resample('YE').last()
        yearly_performance = yearly_data.pct_change().dropna()
        yearly_performance = yearly_performance.rename(index=lambda x: x.year)
        yearly_performance = yearly_performance.sort_index(ascending=False).head(5)
        return yearly_performance




#------------------------------- PAGE CONFIG -------------------------------------------------------#

def style_table(df, width=1000):
    return (df.style.format("{:.1%}")
            .set_properties(**{'color': 'white'})), width

st.set_page_config(page_title= "Investor Reporting App", 
                   layout="wide")

#----------------------------------- IMAGES --------------------------------------------------------#

image_list = ['investor_reporting_app/assets/bar_chart_1.png', 
              'investor_reporting_app/assets/line_chart_1.png',
              'investor_reporting_app/assets/bar_chart_3.png',
              'investor_reporting_app/assets/bar_chart_2.png',
              'investor_reporting_app/assets/histogram.png']


#----------------------------------- SIDEBAR -------------------------------------------------------#

st.sidebar.title('Menu')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file to analyze", type="csv")
st.sidebar.markdown("") # Add space

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

    st.markdown("""
                Investment team can upload a CSV file containing the daily price series or monthly returns of the instruments and choose the desired option from the sidebar menu. 
                
                App will transform data loaded from the file into ready to read format that can be used for on-the-fly analysis or as a part of client reporting.

                ## *No more excel sheets, no more manual calculations.*
                """)
    st.write('---')
    st.stop()


# Radio button to the sidebar
options = ['Monthly Returns', 'Daily Price Series']
choice = st.sidebar.radio("Choose an option", options)
st.sidebar.markdown("")
reporting_date = st.sidebar.date_input('Reporting Date', pd.to_datetime('31/01/2024', format='%d/%m/%Y')) #remove hardcoding
since_inception_date = st.sidebar.date_input('Since Inception Date', pd.to_datetime('21/11/2014', format='%d/%m/%Y'))


#----------------------------------- CONTAINERS ----------------------------------------------------#

# Create a container for the sub-titles
sub_title1_cont = st.container()
sub_title2_cont = st.container()

# Create a row of containers
row1 = st.columns(2)
row2 = st.columns(2)

# Create a new container in the first column and add text to it
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
    data = performance.fund_data.ffill().copy()
    data_display = data.pct_change()
    data_display = data_display + 1
    data_display = data_display.cumprod() - 1
    fig = px.line(data_display, 
                    x=data_display.index, 
                    y=data_display.columns, 
                    labels={'value': '', 'Date': '', 'variable': 'Instrument'},
                    )
    fig.update_layout(title_text="",
                        legend_title_text="",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1,
                            xanchor="right",
                            x=1
                        ),height=500,width= 650)
    fig.update_yaxes(tickformat=".1%")
    return fig, data

#---------------------------- SECOND CONTAINER ---------------------------------#
def create_annualized_performance_container(performance):
    annualized_performance = performance.calculate_annualized_performance().copy()
    annualized_performance_display = annualized_performance.reset_index()
    annualized_performance_display = annualized_performance_display.rename(columns={'index': 'Performance Metric', 'Performance': 'Annualized Performance'})
    annualized_performance_melted = annualized_performance_display.melt(id_vars='Performance Metric', var_name='Instrument', value_name='Annualized Performance')
    fig = px.bar(annualized_performance_melted, x='Instrument', y='Annualized Performance', color='Performance Metric', barmode='group', title='Annualized Performance')
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="")
    fig.update_xaxes(title_text="")
    return fig, annualized_performance

#------------------------------- THIRD CONTAINER -------------------------------#
def create_annualied_volatility_container(performance):
    annualized_volatility = performance.calculate_annualized_volatility().copy()
    annualized_volatility_display = annualized_volatility.reset_index()
    annualized_volatility_display = annualized_volatility_display.rename(columns={'index': 'Performance Metric'})
    annualized_volatility_melted = annualized_volatility_display.melt(id_vars='Performance Metric', var_name='Instrument', value_name='Volatility')
    fig = px.bar(annualized_volatility_melted, x='Instrument', y='Volatility', color='Performance Metric', barmode='group', title='Annualized Volatility')
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
    # Melt the DataFrame    
    calendar_performance_melted = calendar_performance.melt(id_vars='Date', var_name='Instrument', value_name='Performance')
    fig = px.bar(calendar_performance_melted, x='Date', y='Performance', color='Instrument', barmode='group', title='Calendar Performance')
    fig.update_layout(title_text="",legend_title_text='', height=500,width= 650)
    fig.update_yaxes(tickformat=".1%", title_text="")
    fig.update_xaxes(title_text="")
    calendar_performance = calendar_performance.drop(columns='Date')
    return fig, calendar_performance

#------------------------- FIFTH CONTAINER -------------------------------------#
def create_price_chart_container(performance):
    data = performance.fund_data.ffill().copy()
    # Create traces for each instrument
    traces = [go.Scatter(x=data.index, y=data[instrument], name=instrument) for instrument in data.columns]
    # Create a layout without a dropdown menu
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

#----------------------------------- PDF BUTTON ----------------------------------------------------#
#TODO
# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False

# def click_button():
#     st.session_state.clicked = True 

# st.sidebar.button('Save Report as PDF', on_click=click_button)




#----------------------------------- FOOTER --------------------------------------------------------#

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub License](https://img.shields.io/github/license/miroslaw-steblik/investor-reporting-app)]"
    "(https://github.com/miroslaw-steblik/investor-reporting-app)"
)


#----------------------------------------- MAIN ----------------------------------------------------#
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if choice == 'Monthly Returns':  # Monthly Returns option is selected

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
        elif validate_date_existance(since_inception_date, '%d/%m/%Y', data):
            st.error('Since Inception date does not exist. Please choose different date.')
            st.stop()

            
        with sub_title1_cont:
            st.header(':orange[_Monthly Returns_]')


        with first_container:
            fig, data = create_cumulative_performance_container(performance_monthly)
            st.plotly_chart(fig)
            cumulative_performance = performance_monthly.calculate_cumulative_performance()
            st.dataframe(*style_table(cumulative_performance))


        with second_container:
            fig, annualized_performance = create_annualized_performance_container(performance_monthly)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_performance))

        with third_container:
            fig, annualized_volatility = create_annualied_volatility_container(performance_monthly)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_volatility))


        with fourth_container:
            fig, calendar_performance = create_calendar_performance_container(performance_monthly)
            st.plotly_chart(fig)
            calendar_performance = calendar_performance.head(5)
            calendar_performance.index = calendar_performance.index.astype(int).astype(str)
            st.dataframe(*style_table(calendar_performance))
        
        with sixth_container:
            create_histogram_container(performance_monthly)
        
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
            st.dataframe(*style_table(cumulative_performance))


        with second_container:
            fig, annualized_performance = create_annualized_performance_container(performance_daily)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_performance))


        with third_container:
            fig, annualized_volatility = create_annualied_volatility_container(performance_daily)
            st.plotly_chart(fig)
            st.dataframe(*style_table(annualized_volatility))


        with fourth_container:
            fig, calendar_performance = create_calendar_performance_container(performance_daily)
            st.plotly_chart(fig)
            calendar_performance = calendar_performance.head(5)
            calendar_performance.index = calendar_performance.index.astype(int).astype(str)
            st.dataframe(*style_table(calendar_performance))


        if choice == 'Daily Price Series':
            with fifth_container:
                fig = create_price_chart_container(performance_daily)
                st.plotly_chart(fig)
        else:
            pass   

        
        with sixth_container:
            create_histogram_container(performance_daily)

                



            













            
            




