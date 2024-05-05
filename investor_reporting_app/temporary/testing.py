import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from QuantLib import Date, Thirty360 # 30/360 is applied to follow the industry standard for calculating the time period between two dates
from datetime import datetime

monthly_data_path = '/home/miros/DataOps/testing/data/fund_data_26_04_2024.csv'
daily_data_path = '/home/miros/DataOps/testing/data/yahoo_indicies_daily_prices_30_04_2024.csv'
reporting_date = '31/01/2024'
since_inception_date = '31/11/2014'


pd.set_option('display.max_rows', 1500) 

#-------------------------- Decorators ---------------------------------#
# def validate_dates(func):
#     def wrapper(self, *args, **kwargs):
#         if self.reporting_date < self.since_inception_date:
#             raise ValueError("Reporting date cannot be earlier than inception date.")
#         return func(self, *args, **kwargs)
#     return wrapper


def validate_date_format(date, date_format):
    try:
        datetime.strptime(date, date_format)
        return True
    except ValueError:
        return False
    
def validate_reporting_date(reporting_date, since_inception_date):
    if reporting_date < since_inception_date:
        raise ValueError("Reporting date cannot be earlier than inception date.")

def validate_date_range(date, date_format, data):
    date = pd.to_datetime(date, format=date_format)
    min_date = pd.to_datetime(data.index.min(), unit='s')
    max_date = pd.to_datetime(data.index.max(), unit='s')
    if date < min_date or date > max_date:
        raise ValueError("Date is out of data range.")
   

#-------------------------- Monthly Return Series ---------------------------------#

class MonthlyReturnSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = pd.read_csv(monthly_data_path) # data
        self.date_format = '%d/%m/%Y'
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format )
        self.reporting_date = pd.to_datetime(reporting_date, format=self.date_format )
        self.since_inception_date = pd.to_datetime(since_inception_date, format=self.date_format )
        self.fund_data.set_index('Date', inplace=True)

        # Validate dates
        if not validate_date_format(reporting_date, self.date_format) or not validate_date_format(since_inception_date, self.date_format):
            raise ValueError("Invalid date format.")
        validate_reporting_date(self.reporting_date, self.since_inception_date)
        validate_date_range(self.reporting_date, self.date_format, self.fund_data)
        validate_date_range(self.since_inception_date, self.date_format, self.fund_data)
        

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

#-------------------------- Daily Price Series ---------------------------------#
class DailyPriceSeries():
    def __init__(self, data, reporting_date, since_inception_date):
        self.fund_data = pd.read_csv(daily_data_path) # data
        self.date_format = '%d/%m/%Y'
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format )
        self.reporting_date = pd.to_datetime(reporting_date, format=self.date_format )
        self.since_inception_date = pd.to_datetime(since_inception_date, format=self.date_format )
        self.fund_data.set_index('Date', inplace=True)

        # Validate dates
        if not validate_date_format(reporting_date, self.date_format) or not validate_date_format(since_inception_date, self.date_format):
            raise ValueError("Invalid date format.")
        validate_reporting_date(self.reporting_date, self.since_inception_date)
        validate_date_range(self.reporting_date, self.date_format, self.fund_data)
        validate_date_range(self.since_inception_date, self.date_format, self.fund_data)

        # Calculate the period for the since_inception_performance
        start_date = Date(self.since_inception_date.day, self.since_inception_date.month, self.since_inception_date.year)
        end_date = Date(self.reporting_date.day, self.reporting_date.month, self.reporting_date.year)
        since_inception_period = Thirty360(Thirty360.European).yearFraction(start_date, end_date)
        self.periods = {0: 1, 1: 3, 2: 5, 3: 10, 4: since_inception_period}  # The time periods for each performance metric
        self.period_names = {0: '1 year', 1: '3 year', 2: '5 year', 3: '10 year', 4: 'Since Inception'}


    def calculate_cumulative_performance(self):
        self.fund_data = self.fund_data.resample('ME').last()
        one_year_performance = (self.fund_data.loc[self.reporting_date] / self.fund_data.loc[self.reporting_date - relativedelta(years=1)]) -1
        three_year_performance = (self.fund_data.loc[self.reporting_date] / self.fund_data.loc[self.reporting_date - relativedelta(years=3)]) -1
        five_year_performance = (self.fund_data.loc[self.reporting_date] / self.fund_data.loc[self.reporting_date - relativedelta(years=5)]) -1
        ten_year_performance = (self.fund_data.loc[self.reporting_date] / self.fund_data.loc[self.reporting_date - relativedelta(years=10)]) -1
        since_inception_performance = (self.fund_data.loc[self.reporting_date] / self.fund_data.loc[self.since_inception_date]) -1
        performance = pd.concat([one_year_performance, three_year_performance, five_year_performance, ten_year_performance, since_inception_performance], axis=1).T
        performance = performance.rename(index=self.period_names)
        return performance
        # tested and working
    
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
        # Reload daily data
        self.fund_data = pd.read_csv(daily_data_path)
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format)
        self.fund_data.set_index('Date', inplace=True)

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


#---------------------------------- Validation --------------------------------#

#------------------------------ Daily Price Series -----------------------------#
performance_daily = DailyPriceSeries(daily_data_path, reporting_date, since_inception_date)

cumulative_performance = performance_daily.calculate_cumulative_performance()
cumulative_performance = cumulative_performance.map(lambda x: "{:.2%}".format(x))
print()
print('Daily Series -> Cumulative Performance')
print()
print(cumulative_performance)
print()

annualized_performance = performance_daily.calculate_annualized_performance()
annualized_performance = annualized_performance.map(lambda x: "{:.2%}".format(x))
print()
print('Daily Series ->  Annualized Performance')
print()
print(annualized_performance)
print()

annualized_volatility = performance_daily.calculate_annualized_volatility()
annualized_volatility = annualized_volatility.map(lambda x: "{:.2%}".format(x))
print()
print('Daily Series -> Annualized Volatility')
print()
print(annualized_volatility)
print()

calendar_year = performance_daily.calculate_calendar_performance()
calendar_year = calendar_year.map(lambda x: "{:.2%}".format(x))
print()
print('Daily Series -> Calendar Year Performance')
print()
print(calendar_year)
print()

#----------------------------- Validation --------------------------------------#

if since_inception_date not in performance_daily.fund_data.index:
    print()
    print('Daily Series-> Since inception date is not in the data')
fund_data = performance_daily.fund_data.index
data = performance_daily.fund_data

if reporting_date not in fund_data:
    print()
    print('Daily Series -> Reporting date is not in the data')


#------------------------------ Monthly Return Series ---------------------------#

performance_monthly = MonthlyReturnSeries(monthly_data_path, reporting_date, since_inception_date)

cumulative_performance = performance_monthly.calculate_cumulative_performance()
cumulative_performance = cumulative_performance.map(lambda x: "{:.2%}".format(x))
print()
print('Monthly Series -> Cumulative Performance')
print()
print(cumulative_performance)
print()

annualized_performance = performance_monthly.calculate_annualized_performance()
annualized_performance = annualized_performance.map(lambda x: "{:.2%}".format(x))
print()
print('Monthly Series -> Annualized Performance')
print()
print(annualized_performance)
print()

annualized_volatility = performance_monthly.calculate_annualized_volatility()
annualized_volatility = annualized_volatility.map(lambda x: "{:.2%}".format(x))
print()
print('Monthly Series -> Annualized Volatility')
print()
print(annualized_volatility)
print()

calendar_year = performance_monthly.calculate_calendar_performance()
calendar_year = calendar_year.map(lambda x: "{:.2%}".format(x))
print()
print('Monthly Series -> Calendar Year Performance')
print()
print(calendar_year)
print()

#----------------------------- Validation --------------------------------------#

# Check if the Since Inception Date is in the data
since_inception_date_str = performance_monthly.since_inception_date.strftime('%d/%m/%Y')
data_index_str = performance_monthly.fund_data.index.strftime('%d/%m/%Y')
if since_inception_date_str not in data_index_str:
    print()
    print('Monthly Series -> Since inception date is not in the data')


# Check if the Reporting Date is in the data
reporting_date_str = performance_monthly.reporting_date.strftime('%d/%m/%Y')
if reporting_date_str not in data_index_str:
    print()
    print('Monthly Series -> Reporting date is not in the data')



