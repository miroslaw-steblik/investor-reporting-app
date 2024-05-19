import pandas as pd


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