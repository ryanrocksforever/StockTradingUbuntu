import quandl
import warnings
import os
from bulbea._util.color import Color
from bulbea.config.app import AppConfig
from bulbea._util import (
    _check_type,
    _check_str,
    _check_int,
    _check_pandas_series,
    _check_pandas_dataframe,
    _check_iterable,
    _check_environment_variable_set,
    _validate_date,
    _assign_if_none,
    _get_type_name,
    _get_datetime_str,
    _raise_type_error,
    _is_sequence_all
)
from bulbea._util.const import (
    ABSURL_QUANDL,
    QUANDL_MAX_DAILY_CALLS,
    SHARE_ACCEPTED_SAVE_FORMATS
)
envvar = AppConfig.ENVIRONMENT_VARIABLE['quandl_api_key']
print(envvar)


quandl.ApiConfig.api_key = os.getenv(envvar)
def getData(source, ticker):
    data    = quandl.get('{database}/{code}'.format(
        database = source,
        code     = ticker
    ), start_date="2018-1-20", end_date="2019-04-28")
    return  data

data = getData("WIKI", "AAPL")
print(data)