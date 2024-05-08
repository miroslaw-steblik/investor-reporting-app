
""" MIT License, Copyright (C) 2024, Miroslaw Steblik """

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dateutil.relativedelta import relativedelta
from QuantLib import Date, Thirty360 # 30/360 is applied to follow the industry standard for calculating the time period between two dates
import plotly.graph_objects as go
import colorlover as cl


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


#----------------------------- MONTHLY RETURN CLASS ------------------------------------------------#
class MonthlyReturnSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = data
        self.date_format = '%d/%m/%Y'
        self.fund_data = self.load_data()

        self.reporting_date = self.validate_date(reporting_date, 'reporting_date')
        self.since_inception_date = self.validate_date(since_inception_date, 'since_inception_date')
        if self.since_inception_date > self.reporting_date:
            raise ValueError('since_inception_date must be earlier than reporting_date')

        
        # Calculate the period for the since_inception_performance
        start_date = Date(self.since_inception_date.day, self.since_inception_date.month, self.since_inception_date.year)
        end_date = Date(self.reporting_date.day, self.reporting_date.month, self.reporting_date.year)
        since_inception_period = Thirty360(Thirty360.European).yearFraction(start_date, end_date)
        self.periods = {0: 1, 1: 3, 2: 5, 3: 10, 4: since_inception_period}  # The time periods for each performance metric
        self.period_names = {0: '1 year', 1: '3 year', 2: '5 year', 3: '10 year', 4: 'Since Inception'}

    def load_data(self):
        data = self.fund_data
        data['Date'] = pd.to_datetime(data['Date'], format=self.date_format)
        data.set_index('Date', inplace=True)
        return data
    
    def validate_date(self, date_str, date_name):
        try:
            date = pd.to_datetime(date_str, format='%d/%m/%Y')
        except ValueError:
            raise ValueError(f'Invalid date format for {date_name}. Expected format: dd/mm/yyyy')
        return date
    
    @property
    def one_year_period(self):
        return self.check_period(self.reporting_date - pd.DateOffset(months=11), self.reporting_date)
    @property
    def three_year_period(self):
        return self.check_period(self.reporting_date - pd.DateOffset(months=35), self.reporting_date)
    @property
    def five_year_period(self):
        return self.check_period(self.reporting_date - pd.DateOffset(months=59), self.reporting_date)
    @property
    def ten_year_period(self):
        return self.check_period(self.reporting_date - pd.DateOffset(months=119), self.reporting_date)
    @property
    def since_inception_period(self):
        return self.check_period(self.since_inception_date, self.reporting_date)
    

    def check_period(self, start_date, end_date):
        period = self.fund_data.loc[(self.fund_data.index >= start_date) & (self.fund_data.index <= end_date)]
        expected_months = pd.date_range(start=start_date, end=end_date, freq='ME').shape[0]
        if period.shape[0] == expected_months:
            return period
        else:
            return None
        ### COMPLETED

    def calculate_cumulative_performance(self):
        def calculate_performance(period):
            if period is None:
                return None
            return (period + 1).prod() - 1

        one_year_performance = calculate_performance(self.one_year_period)
        three_year_performance = calculate_performance(self.three_year_period)
        five_year_performance = calculate_performance(self.five_year_period)
        ten_year_performance = calculate_performance(self.ten_year_period)
        since_inception_performance = calculate_performance(self.since_inception_period)
        performance = pd.concat([pd.Series(one_year_performance, name='1 year'),
                                pd.Series(three_year_performance, name='3 year'),
                                pd.Series(five_year_performance, name='5 year'),
                                pd.Series(ten_year_performance, name='10 year'),
                                pd.Series(since_inception_performance, name='Since Inception')], axis=1).T
        performance = performance.rename(index=self.period_names)
        return performance
        ### COMPLETED

    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}     # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        ### COMPLETED

    def calculate_annualized_volatility(self):
        def calculate_volatility(period):
            if period is None:
                return None
            return period.std(ddof=0) * np.sqrt(12)

        volatility_fund_data = data
        volatility_fund_data = volatility_fund_data.resample('ME').last()
        one_year_volatility = calculate_volatility(self.one_year_period)
        three_year_volatility = calculate_volatility(self.three_year_period)
        five_year_volatility = calculate_volatility(self.five_year_period)
        ten_year_volatility = calculate_volatility(self.ten_year_period)
        since_inception_volatility = calculate_volatility(self.since_inception_period)
        volatility = pd.concat([pd.Series(one_year_volatility, name='1 year'),
                                pd.Series(three_year_volatility, name='3 year'),
                                pd.Series(five_year_volatility, name='5 year'),
                                pd.Series(ten_year_volatility, name='10 year'),
                                pd.Series(since_inception_volatility, name='Since Inception')], axis=1).T
        volatility = volatility.rename(index=self.period_names)
        return volatility
        ### COMPLETED

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
        ### COMPLETED

#----------------------------- Daily Price Series Class --------------------------------------------#
class DailyPriceSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = data
        self.date_format = '%d/%m/%Y'
        self.fund_data = self.load_data()

        self.reporting_date = self.validate_date(reporting_date, 'reporting_date')
        self.since_inception_date = self.validate_date(since_inception_date, 'since_inception_date')
        if self.since_inception_date > self.reporting_date:
            raise ValueError('since_inception_date must be earlier than reporting_date')
        
        start_date = Date(self.since_inception_date.day, self.since_inception_date.month, self.since_inception_date.year)
        end_date = Date(self.reporting_date.day, self.reporting_date.month, self.reporting_date.year)
        since_inception_period = Thirty360(Thirty360.European).yearFraction(start_date, end_date)
        self.periods = {0: 1, 1: 3, 2: 5, 3: 10, 4: since_inception_period}  # The time periods for each performance metric
        self.period_names = {0: '1 year', 1: '3 year', 2: '5 year', 3: '10 year', 4: 'Since Inception'}


    def load_data(self):
        data = self.fund_data
        data['Date'] = pd.to_datetime(data['Date'], format=self.date_format)
        data.set_index('Date', inplace=True)
        return data
    
    def validate_date(self, date_str, date_name):
        try:
            date = pd.to_datetime(date_str, format='%d/%m/%Y')
        except ValueError:
            raise ValueError(f'Invalid date format for {date_name}. Expected format: dd/mm/yyyy')
        return date

    def calculate_cumulative_performance(self):
        daily_fund_data = self.fund_data.asfreq('D').ffill()  # Use daily data
        monthly_fund_data = self.fund_data.resample('ME').last()  # Use month-end data

        def calculate_performance(end_date, start_date):
            try:
                end_value = monthly_fund_data.loc[end_date]
                start_value = daily_fund_data.loc[start_date]
                return (end_value / start_value) - 1
            except KeyError:
                return None

        since_inception_performance = pd.Series(calculate_performance(self.reporting_date, self.since_inception_date), name='Since Inception')
        one_year_performance = pd.Series(calculate_performance(self.reporting_date, self.reporting_date - relativedelta(years=1)), name='1 year')
        three_year_performance = pd.Series(calculate_performance(self.reporting_date, self.reporting_date - relativedelta(years=3)), name='3 year')
        five_year_performance = pd.Series(calculate_performance(self.reporting_date, self.reporting_date - relativedelta(years=5)), name='5 year')
        ten_year_performance = pd.Series(calculate_performance(self.reporting_date, self.reporting_date - relativedelta(years=10)), name='10 year')
        performance = pd.concat([one_year_performance, three_year_performance, five_year_performance, ten_year_performance,since_inception_performance], axis=1).T
        performance = performance.rename(index=self.period_names)
        return performance
        ### COMPLETED

    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}      # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        ### COMPLETED


    def calculate_volatility(self, period):
        if period is None or period.empty:
            return None
        return period.std(ddof=0) * np.sqrt(252)

    def get_period(self, data, start_date, end_date):
        if start_date < data.index[0]:
            return None
        actual_start_date = data.index.asof(start_date)
        return data.loc[actual_start_date:end_date]

    def calculate_annualized_volatility(self):
        # Reload the data
        self.fund_data = data
        self.fund_data = self.fund_data.ffill()
        returns = self.fund_data.pct_change()
        # Get periods
        one_year_period = self.get_period(returns, self.reporting_date - pd.DateOffset(years=1, days=-1), self.reporting_date)
        three_year_period = self.get_period(returns, self.reporting_date - pd.DateOffset(years=3, days=-1), self.reporting_date)
        five_year_period = self.get_period(returns, self.reporting_date - pd.DateOffset(years=5, days=-1), self.reporting_date)
        ten_year_period = self.get_period(returns, self.reporting_date - pd.DateOffset(years=10, days=-1), self.reporting_date)
        since_inception_period = self.get_period(returns, self.since_inception_date, self.reporting_date)

        one_year_volatility = self.calculate_volatility(one_year_period)
        three_year_volatility = self.calculate_volatility(three_year_period)
        five_year_volatility = self.calculate_volatility(five_year_period)
        ten_year_volatility = self.calculate_volatility(ten_year_period)
        since_inception_volatility = self.calculate_volatility(since_inception_period)

        volatility = pd.concat([pd.Series(one_year_volatility, name='1 year'),
                                pd.Series(three_year_volatility, name='3 year'),
                                pd.Series(five_year_volatility, name='5 year'),
                                pd.Series(ten_year_volatility, name='10 year'),
                                pd.Series(since_inception_volatility, name='Since Inception')], axis=1).T
        volatility = volatility.rename(index=self.period_names)
        return volatility
        ### COMPLETED

        
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
        ### COMPLETED

#------------------------------- PAGE CONFIG -------------------------------------------------------#
def style_table(df, width=1000):
    return (df.style.format("{:.1%}")
            .set_properties(**{'color': 'white'})), width

st.set_page_config(page_title= "Investor Reporting App", layout="wide")

#----------------------------------- IMAGES --------------------------------------------------------#
image_list = ['investor_reporting_app/assets/bar_chart_1.png', 
              'investor_reporting_app/assets/line_chart_1.png',
              'investor_reporting_app/assets/bar_chart_3.png',
              'investor_reporting_app/assets/bar_chart_2.png',
              'investor_reporting_app/assets/histogram.png']

#----------------------------------- SIDEBAR -------------------------------------------------------#
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

# Radio button to the sidebar
options = ['Monthly Returns', 'Daily Price Series']
choice = st.sidebar.radio("Choose an option", options)
st.sidebar.markdown("")
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
          
            with sixth_container:
                create_histogram_container(performance_daily)




                



            













            
            




