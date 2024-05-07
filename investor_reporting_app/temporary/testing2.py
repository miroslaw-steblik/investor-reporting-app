import pandas as pd
from dateutil.relativedelta import relativedelta

reporting_date = '2023-09-30'



dates = [reporting_date - relativedelta(years=1, days=-1):reporting_date]
print(dates)