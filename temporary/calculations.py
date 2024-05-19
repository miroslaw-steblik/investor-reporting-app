import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from QuantLib import Date, Thirty360 # 30/360 is applied to follow the industry standard for calculating the time period between two dates
from datetime import datetime

monthly_data_path = '/home/miros/DataOps/testing/data/monthly_returns_less_then_10_years.csv'
daily_data_path = '/home/miros/DataOps/testing/data/daily_prices_less_then_10_years.csv'
reporting_date = '30/04/2023'
since_inception_date = '25/09/2015'


pd.set_option('display.max_rows', 100) 


#-------------------------- Validation Functions ---------------------------------#
def validate_date_format(date, date_format):
    try:
        datetime.strptime(date, date_format)
        return True
    except ValueError:
        return False
    

def validate_date_range(date, date_format, data):
    date = pd.to_datetime(date, format=date_format)
    min_date = pd.to_datetime(data.index.min(), unit='s')
    max_date = pd.to_datetime(data.index.max(), unit='s')
    if date < min_date or date > max_date:
        raise ValueError("Date is out of data range.")
   

#-------------------------- Monthly Return Series ---------------------------------#
class MonthlyReturnSeries():
    def __init__(self, data_path, reporting_date, since_inception_date):
        self.data_path = data_path
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
        data = pd.read_csv(monthly_data_path)
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


    # check if the data is available for the given period in #mask
    def check_period(self, start_date, end_date):
        period = self.fund_data.loc[(self.fund_data.index >= start_date) & (self.fund_data.index <= end_date)]
        expected_months = pd.date_range(start=start_date, end=end_date, freq='ME').shape[0]
        if period.shape[0] == expected_months:
            return period
        else:
            return None
        ### CODE WORKING

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
        ### CODE WORKING

    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}     # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        ### CODE WORKING

    def calculate_annualized_volatility(self):
        def calculate_volatility(period):
            if period is None:
                return None
            return period.std(ddof=0) * np.sqrt(12)

        volatility_fund_data = pd.read_csv(monthly_data_path)  #data
        volatility_fund_data['Date'] = pd.to_datetime(volatility_fund_data['Date'], format=self.date_format )
        volatility_fund_data.set_index('Date', inplace=True)
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
        ### CODE WORKING

    def calculate_calendar_performance(self):
        yearly_data = self.fund_data.resample('YE')
        performance_list = [(data + 1).prod() - 1 for year, data in yearly_data if len(data) == 12]
        dates_list = [year.year for year, data in yearly_data if len(data) == 12]
        performance_df = pd.DataFrame(performance_list, index=dates_list, columns=self.fund_data.columns)
        performance_df.sort_index(ascending=False, inplace=True)
        return performance_df.head(5)
        ### CODE WORKING

#----------------------------- Daily Price Series Class --------------------------------------------#
class DailyPriceSeries():
    def __init__(self, data_path, reporting_date, since_inception_date):
        self.data_path = data_path
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
        data = pd.read_csv(daily_data_path)
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
        ###CODE WORKING

    
    def calculate_annualized_performance(self):
        def annualized_return(r, t):
            return (1 + r) ** (1 / t) - 1

        cumulative_performance = self.calculate_cumulative_performance()
        cumulative_performance = cumulative_performance.rename(index=self.period_names)
        reverse_period_names = {v: k for k, v in self.period_names.items()}      # Create a reverse mapping dictionary
        annualized_performance = cumulative_performance.apply(lambda r: annualized_return(r, self.periods[reverse_period_names[r.name]]), axis=1)
        return annualized_performance
        ### CODE WORKING


    def calculate_volatility(self, period):
        if period is None or period.empty:
            return None
        return period.std(ddof=0) * np.sqrt(252)

    def get_period(self, data, start_date, end_date):
        if start_date < data.index[0]:
            return None
        actual_start_date = data.index.asof(start_date)
        return data.loc[actual_start_date:end_date]

    def calculate_volatility_for_periods(self):
        # Reload the data
        self.fund_data = pd.read_csv(daily_data_path)
        self.fund_data['Date'] = pd.to_datetime(self.fund_data['Date'], format=self.date_format)
        self.fund_data.set_index('Date', inplace=True)

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
        ### CODE WORKING



    def calculate_calendar_performance(self):
        monthly_data = self.fund_data.resample('ME').last()
        years_with_12_months = monthly_data.groupby(monthly_data.index.year).size() == 12
        valid_years = years_with_12_months[years_with_12_months].index
        valid_data = self.fund_data[self.fund_data.index.year.isin(valid_years)]  # Filter the data by the valid years
        yearly_performance = (valid_data
                            .resample('YE')
                            .last()
                            .pct_change()
                            .dropna())
        yearly_performance = yearly_performance.rename(index=lambda x: x.year)
        yearly_performance = yearly_performance.sort_index(ascending=False)
        return yearly_performance.head(5)
        ### CODE WORKING


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

annualized_volatility = performance_daily.calculate_volatility_for_periods()
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




