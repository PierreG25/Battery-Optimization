import pandas as pd

year = True

if year:
    file_path = 'data/raw_spot_price_2025.csv'
else:
    file_path = 'data/raw_spot_price_01:11:2025.csv'

def clean(df, datetime_col='MTU (CET/CEST)'):
    df = df[df['Sequence'] == 'Without Sequence']
    df = df.drop(columns=['Area', 'Sequence', 'Intraday Period (CET/CEST)', 'Intraday Price (EUR/MWh)'])
    df[datetime_col] = df[datetime_col].str.split(' - ').str[0]
    print('CONVERTING DATES')
    print(df)
    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, format='mixed')
    if year:
        df = df[df[datetime_col] >= '01/12/2025']
    print('ok')

    df.rename(columns={'Day-ahead Price (EUR/MWh)' : 'price',
                    'MTU (CET/CEST)' : 'time'}, 
                    inplace=True)

    df.to_csv('data/clean_spot_price_2025.csv', index=False)

    return df

def hourly_sample(df, datetime_col='time'):
    df = df.set_index(datetime_col)
    df = df.resample('h').first().reset_index()

    df.to_csv('data/clean_hourly_spot_price_2025.csv', index=False)

df = pd.read_csv(file_path)
cleaned_df = clean(df)
hourly_sample(cleaned_df)
