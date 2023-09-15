# blockchain-competitive-analysis

# Code Documentation

This documentation provides an overview of the Python code you've provided. The code appears to be related to data analysis and visualization, specifically for analyzing blockchain data from various chains like Arbitrum, Optimism, and Polygon. Below, I'll break down the code into sections and provide explanations for each part.

### Importing Libraries

The code begins by importing several Python libraries that are commonly used for data analysis and visualization. These libraries include Pandas, NumPy, Seaborn, Matplotlib, Datetime, Glob, Plotly, Scipy, OS, Requests, tqdm, and dotenv. Here's a brief explanation of their usage:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import glob
import matplotlib.dates as mdates
import plotly.express as px 
import matplotlib
import matplotlib.gridspec as gridspec
import plotly.offline as pyo
from scipy import stats
import os
import requests
from tqdm import tqdm
import plotly.io as pio
```

These libraries are used for data manipulation, visualization, and making HTTP requests. `dotenv` is used for loading environment variables, and `tqdm` is used for displaying progress bars.

### Setting Default Plotly Renderer

```python
pio.renderers.default = 'notebook'
```

This line sets the default renderer for Plotly charts to 'notebook'.

### Data Cleaning Functions

The code defines several functions for cleaning and preprocessing data:

- `clean_timestamp_transpose`: Cleans up a timestamp column to have the format 'year-month-day'.
- `clean_timestamp_dune`: Cleans up a timestamp column in a different format.
- `string_to_datetime`: Converts a string to a datetime object with the format 'year-month-day'.
- `clean_raw_dune`: Performs various data cleaning operations on a DataFrame, including sorting by date and time, removing duplicate transactions, and cleaning the timestamp.
- `agg_dfs`: Aggregates multiple DataFrames by concatenating them.
- `addlabels`: A function to add labels to a bar chart.

### Dune API Usage

The code includes usage of the Dune API to retrieve data. It sets up a query using the DuneClient library and makes a request to the Dune API to fetch data.

### Plotting Functions

There are several plotting functions defined in the code. These functions create various types of visualizations for data analysis. Here are some of them:

- `plot_contractsdistribution`: Plots the distribution of contracts and events over time.
- `plot_protocolmarketshare`: Plots the transaction share of protocols, contracts, and events.
- `most_popular_contracts`: Returns and prints the most popular contracts.
- `most_popular_events`: Returns and prints the most popular events.

This documentation provides an overview of the Python code you've provided for data preprocessing and analysis of blockchain data from different chains, including Polygon, Arbitrum, zkSync Era, and Optimism. Below, I'll break down the code into sections and provide explanations for each part.

## Polygon Data Preprocessing

### Concatenating Data
The code starts by concatenating data from multiple CSV files into a single DataFrame called `poly`. These CSV files appear to contain Polygon blockchain data for different time periods.

```python
poly = pd.concat([temp1, temp2, temp3, temp4, temp5, temp6, temp7], ignore_index=True)
```

### Timestamp Cleaning
The `poly` DataFrame's "timestamp" column is cleaned and standardized to have the format "year-month-day."

```python
poly['timestamp'] = poly['timestamp'].map(clean_timestamp_transpose)
poly['timestamp'] = pd.to_datetime(poly['timestamp'], format="%Y-%m-%d")
```

### Date Column Creation
A new column called "date" is created in the `poly` DataFrame, containing only the month and day information from the "timestamp" column.

```python
def month_day(t):
    return f"{t.month}-{t.day}"

poly['date'] = poly['timestamp'].map(month_day)
```

### Removing Duplicate Transactions and Sorting
Duplicate transactions are removed based on the "transaction_hash" column, and the DataFrame is sorted by the "timestamp."

```python
poly = poly.drop_duplicates(subset=['transaction_hash'], keep='first')
poly.sort_values(by='timestamp', inplace=True)
```

## Polygon Data Analysis

### Transaction Count Aggregation
The code aggregates the transaction count by day and stores the result in the `txs` DataFrame.

```python
txs = poly.groupby(poly['timestamp']).agg({'count'})
```

### Loading Additional Polygon Transaction Data
Additional transaction data is loaded from a CSV file into the `poly_txs` DataFrame.

```python
poly_txs = pd.read_csv('./Chains Data/polygon txs/polygon_txs.csv')
```

### Date Conversion
The "Date(UTC)" column in the `poly_txs` DataFrame is converted to a datetime format.

```python
poly_txs['Date(UTC)'] = pd.to_datetime(poly_txs['Date(UTC)'], format="%m/%d/%Y")
```

### Filtering Data
Data in the `poly_txs` DataFrame is filtered to include only rows with dates before September 8, 2020.

```python
poly_txs100 = poly_txs[poly_txs['Date(UTC)'] < datetime.datetime(2020, 9, 8)]
```

### Transaction Count Plot
A time-series chart is created to visualize the number of Polygon transactions per day over time.

```python
plt.figure(1)
plt.plot(poly_txs100['Date(UTC)'], poly_txs100['Value'])
plt.xticks(rotation=40)
plt.title('Polygon transactions per day')
```

The above code is followed by similar sections for other blockchain chains, including Arbitrum, zkSync Era, and Optimism, where data preprocessing and analysis are performed. Each section loads, preprocesses, and analyzes data from the respective blockchain chain.


```

This documentation provides an overview of the functions and analyses conducted on user acquisition and retention for Arbitrum and Optimism blockchain chains.
```

```markdown
# Acquisition: How Many Users Does Each Protocol Bring?

## Functions

### `first_time_users_protocol(protocol_df)`

Input the dataframe containing the blockchain's protocols' data and return a dataframe containing how many first-time users each protocol attracted to the chain.

### `plot_first_time_users_protocol(df, df_pct, topn, chain_name)`

Plot the number of first-time users each protocol attracted to the chain.

## Arbitrum

### Number of First-Time Users Each Protocol Attracted to Arbitrum

```python
#Number of first time users each protocol attracted to Arbitrum
first_time_users_per_protocol_arb, first_time_users_per_protocol_pct_arb = first_time_users_protocol(protocol_arb)
```

### Plot Those First-Time Users

```python
#Plot those first time users
plot_first_time_users_protocol(first_time_users_per_protocol_arb, first_time_users_per_protocol_pct_arb, topn=20, chain_name='Arbitrum')
```

## Optimism

### Number of First-Time Users Each Protocol Attracted to Optimism

```python
#Number of first time users each protocol attracted to Arbitrum
first_time_users_per_protocol_op, first_time_users_per_protocol_pct_op = first_time_users_protocol(protocols_op)
```

### Plot Those First-Time Users Per Protocol

```python
#Plot those first time users per protocol
plot_first_time_users_protocol(first_time_users_per_protocol_op, first_time_users_per_protocol_pct_op, 20, 'Optimism')
```

# User Retention (Average Lifespan Users)

## Functions

### `calculate_lifespan(user, df)`

Calculate the lifespan of a user.

### `users_lifespan_per_protocol(protocol, users_by_protocol, protocols_df, sample_size=385)`

Calculate the lifespan of each user attracted to the chain by each respective protocol.

### `avg_users_lifespan_by_protocol(protocols_df, topn=20)`

Return the mean, median, and standard deviation of each protocol's users' lifespan.

### `plot_users_lifespan_by_protocol(df, chain_name)`

Plot each protocol's user lifespan, mean, median, and standard deviation.

### `plot_avg_users_lifespan_by_protocol(df, chain_name, ordered=False)`

Plot just the mean user lifespan.

### `plot_avg_users_lifespan_by_protocol_sns(df, chain_name, ordered=False)`

Same as the function above but use seaborn library for a different style.

### `chain_user_freq_t10(protocol_df)`

Number of one-time, two-time, etc. all the way to ten-time users on a chain.

### `chain_user_freq(protocol_df, topn)`

Number of one-time, two-time, etc. users on a chain, you can specify how many X-time users to calculate up to.

### `chain_user_freq_pct(protocol_df, topn)`

Percentage of total users are one-time, two-time, three-time, etc. users on a chain.

### `plot_chain_user_freq(pop_df, title)`

Plot number of one-time,... users.

### `plot_chain_user_freq_topn(protocol_df, topn, title)`

Plot number of one-time,... users, specifying X time users.

### `plot_chain_user_freq_topn_pct(protocol_df, topn, title)`

Plot percentage of total users who are one-time, etc. users, specifying X time users.
```
Here's a GitHub documentation in markdown format for your code:

```markdown
# Protocol User Retention

## Functions

### `userretention_mom(protocol_df, app='Dune')`

For an interval of every month, calculate the percentage of users that returned.

- `protocol_df`: DataFrame containing the blockchain's protocols' data.
- `app`: Application identifier ('Dune' or 'Transpose'). Default is 'Dune'.

### `plot_userretention(ur_df, title, calendar_dts=True)`

Plot user retention month-over-month.

- `ur_df`: User retention DataFrame.
- `title`: Title for the plot.
- `calendar_dts`: Boolean to specify if month labels should be calendar dates (True) or sequential numbers (False). Default is True.

## Polygon

### User Retention Calculation for Polygon

```python
userretention_poly = userretention_mom(poly, app='Transpose')
```

### Plot User Retention for Polygon

```python
plot_userretention(userretention_poly, 'Polygon')
```

## Arbitrum

### User Retention Calculation for Arbitrum

```python
userretention_arb = userretention_mom(all_arb, app='Transpose')
```

### Plot User Retention for Arbitrum

```python
plot_userretention(userretention_arb, title='Arbitrum')
```

## zkSync

### User Retention Calculation for zkSync

```python
userretention_zksync = userretention_mom(all_zksync)
```

### Plot User Retention for zkSync

```python
plot_userretention(userretention_zksync, title='zkSync')
```

## Optimism

### User Retention Calculation for Optimism

```python
userretention_op = userretention_mom(all_op)
```

### Plot User Retention for Optimism

```python
plot_userretention(userretention_op, title='Optimism')
```
Here's a GitHub documentation in markdown format for your code:

```markdown
# Uniswap Data on Optimism and Arbitrum

## Optimism

### Data Import and Preprocessing

```python
#contracts
path = 'Chains Data/optimism/uniswap'
uniswap_op = pd.read_csv(path + 'uniswap_op.csv')
uniswap_op['block_time'] = uniswap_op['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
```

### Activity

```python
uniswap_op_txs = protocol_txs(uniswap_op)
plot_protocoltxs(uniswap_op_txs, 'Uniswap (Optimism)', datetime.datetime(2021, 12, 16))
```

### Most Used Contracts

```python
plot_contractsdistribution(uniswap_op, 'Uniswap', 'Optimism')
```

### Unique Addresses

```python
uniswap_op_uniqueaddresses = protocol_uniqueaddresses(uniswap_op)
plot_uniqueaddresses(uniswap_op_uniqueaddresses, 'Uniswap (Optimism)')
```

### User Retention

```python
userretention_pct = userretention_mom(uniswap_op)
plot_userretention(userretention_pct, 'Uniswap (Optimism)', calendar_dts=True)
```

### Protocol Popularity

```python
protpopt10_op = protocol_popularity_t10(uniswap_op)
protpop_op = protocol_popularity(uniswap_op, 100)

plot_protocolpopularity(protpopt10_op, 'Uniswap (Optimism)')
plot_protocolpopularity(protpop_op, 'Uniswap (Optimism)')
```

## Arbitrum

### Data Import and Preprocessing

```python
#contracts
path = 'Chains Data/arbitrum txs/uniswap'
uniswap_arb = pd.read_csv(path + 'uniswap_arb.csv')
uniswap_arb['block_time'] = uniswap_arb['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
```

### Activity

```python
uniswap_txs_arb = protocol_txs(uniswap_arb)
uniswap_txs_arb = uniswap_txs_arb.iloc[:-1]
plot_protocoltxs(uniswap_txs_arb, 'Uniswap (Arbitrum)', datetime.datetime(2021, 8, 31))
```

### Most Used Contracts

```python
plot_contractsdistribution(uniswap_arb, 'Uniswap', 'Arbitrum')
```

### Unique Addresses

```python
uniswap_unique_arb = protocol_uniqueaddresses(uniswap_arb)
plot_uniqueaddresses(uniswap_unique_arb, 'Uniswap (Arbitrum)')
```

### User Retention

```python
userretention_pct = userretention_mom(uniswap_arb)
plot_userretention(userretention_pct, 'Uniswap (Arbitrum)', calendar_dts=True)
```

### Protocol Popularity

```python
protpopt10_arb = protocol_popularity_t10(uniswap_arb)
protpop_arb = protocol_popularity(uniswap_arb, 100)

plot_protocolpopularity(protpopt10_arb, 'Uniswap (Arbitrum)')
plot_protocolpopularity(protpop_arb, 'Uniswap (Arbitrum)')
```


