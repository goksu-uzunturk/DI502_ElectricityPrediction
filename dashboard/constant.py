'''list_of_models = ['Linear_Regression','Decision Tree', 'Moving Average', 'XGBoost', 'ARIMA']


FORECASTS_PATH = 'https://raw.githubusercontent.com/kbaranko/peaky-finders/master/peaky_finders/forecasts'
"""Github forecast csv files."""'''


MA_MODEL_DESCRIPTION = '''
    The Moving Average forecasting was trained on historical meter readings, weather, and building data
    from 2016-2017 to design a baseline. Temperature readings are from site_id - 1 and site_id - 6.
'''


DecisionTree_MODEL_DESCRIPTION = '''
    The Decision Tree forecasting model was trained on historical meter readings, weather, and building data
    from 2016-2017. Temperature readings are from site_id - 1 and site_id - 6.
'''


XGBoost_MODEL_DESCRIPTION = '''
    The XG Boost forecasting model was trained on historical meter readings, weather, and building data
    from 2016-2017. Temperature readings are from site_id - 1 and site_id - 6.
'''
Dataset_DESCRIPTION = '''
    The dataset is taken from [ASHRAE Great Energy Predictor III Competition](https://www.kaggle.com/c/ashrae-energy-prediction) 
'''
ARIMA_DESCRIPTION = '''
    The ARIMA forecasting model was trained on historical meter readings, weather, and building data
    from 2016-2017. Temperature readings are from site_id - 1 and site_id - 6.
'''



