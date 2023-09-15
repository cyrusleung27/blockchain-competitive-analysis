# blockchain-competitive-analysis

This repo contains the code and data necessary to analyze high-level metrics like transactions per day, users per day, average gas price per day, and more, of 4 blockchains — Arbitrum, Optimism, Polygon, and zkSync Era — while also allowing you to visualize the correlations between these parameters. 

This code also allows you to analyze and visualize the top protocols on these chains, synthesize their user acquisition and retention successes -> all of which are vitally insightful to go-to-market business development strategy decisions.

Specific metrics regarding Uniswap, GMX dex, Level Finance, and more, are available.

*Please reconfigure the path variables in the Python script for your personal use.

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

```markdown
# Level Finance (on BNB)

## Data Retrieval and Preprocessing

```python
# Get the protocol Level Finance's data and perform data aggregation and cleaning
path = 'Chains Data/level_finance'

level = agg_dfs(path)
level = clean_raw_dune(level)
```

## Transactions

```python
# Calculate and plot transactions per day for Level Finance on BNB
level_txs = protocol_txs(level)
plot_protocoltxs(level_txs, 'Level Finance (BNB)', datetime.datetime(2022, 12, 26))
```

## Most Used Contracts

```python
# Plot the distribution of most used contracts for Level Finance on BNB
plot_contractsdistribution(level, 'Level Finance', 'BNB')
```

## Number of Unique Addresses per Day

```python
# Calculate and plot the number of unique addresses per day for Level Finance on BNB
level_unique = protocol_uniqueaddresses(level)
plot_uniqueaddresses(level_unique, 'Level Finance (BNB)')
```

# GMX (on Arbitrum)

## Data Retrieval and Preprocessing

```python
# Get the protocol GMX's data and perform data retrieval and preprocessing
path = 'Chains Data/gmx'

gmx = pd.read_csv(path + '/gmx.csv')
gmx = gmx.drop(['value', 'gas_limit', 'effective_gas_price'], axis=1)
gmx['block_time'] = gmx['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
```

## Transactions

```python
# Calculate and plot transactions per day for GMX on Arbitrum
gmx_txs = protocol_txs(gmx)
plot_protocoltxs(gmx_txs, 'GMX (Arbitrum)', datetime.datetime(2021, 9, 1))
```

## Most Used Contracts

```python
# Plot the distribution of most used contracts for GMX on Arbitrum
plot_contractsdistribution(gmx, 'GMX', 'Arbitrum')
```

## Number of Unique Addresses per Day

```python
# Calculate and plot the number of unique addresses per day for GMX on Arbitrum
gmx_unique = protocol_uniqueaddresses(gmx)
plot_uniqueaddresses(gmx_unique, 'GMX (Arbitrum)')
```

```markdown
# Get Transactions Per Day Chart for zkSync, Optimism, Arbitrum, and Polygon

## Overview

This code snippet retrieves transaction data for zkSync, Optimism, Arbitrum, and Polygon, and plots transactions per day for each of these blockchain networks.

## Code Snippet

The code consists of several sections for data retrieval, data preprocessing, and visualization. Below is a summary of each section:

### zkSync Transactions

```python
# Retrieve zkSync transaction data and aggregate transactions per day
zksync_txs = all_zksync.groupby(all_zksync['block_time']).agg({'count'})
zksync_txs = zksync_txs[zksync_txs.index < datetime.datetime(2023, 7, 3)]
```

### Optimism Transactions

```python
# Retrieve Optimism transaction data and aggregate transactions per day
txs_op = all_op.groupby(all_op['block_time']).agg({'count'})
```

### Polygon and Arbitrum Transactions (Visualization)

The code includes visualization of transactions for Polygon and Arbitrum for the first 100 days after their respective launches.

### Comparison of Transactions

```python
# Comparison of transactions per day for Polygon, Arbitrum, zkSync, and Optimism
compare_txs = pd.DataFrame(columns=['Day', 'Polygon_txs', 'Arbitrum_txs', 'zkSync_txs', 'Optimism_txs'])
# ...
```

### Normalized Comparison of Transactions

```python
# Normalized comparison of transactions per day for Polygon, Arbitrum, zkSync, and Optimism
compare_txs_norm = pd.DataFrame(columns=['Day', 'Polygon_txs', 'Arbitrum_txs', 'zkSync_txs', 'Optimism_txs'])
# ...
```

### Analysis of Transactions

The code includes analysis sections for zkSync Era and Polygon transactions.

### Number of Smart Contracts Created

The code analyzes the number of smart contracts created on Arbitrum and Polygon and visualizes the data.

## Usage

You can use this code to retrieve and analyze transaction data for zkSync, Optimism, Arbitrum, and Polygon. Additionally, it provides visualizations to help you understand transaction trends and smart contract creations on these blockchain networks.

Feel free to adapt and extend this code as needed for your specific analysis and research.

```

```markdown
# Address Activity Distribution

## Is there a correlation between gas price and transactions per day?

### Code Snippet
```python
zksync_txs
```

### Function to Create Correlation Data Frame
```python
def create_corr_df(chain):
    '''
    Create a correlation heatmap for the various variables in a blockchain's transaction data
    '''
    temp_corr = pd.DataFrame(columns=['block_time','txs','gas_price','unique_addresses','contracts_deployed'])
    dates = np.unique([d for d in chain['block_time']])
    temp_corr['block_time'] = dates
    
    #txs
    temp_txs = chain.copy()
    temp_txs = temp_txs.groupby(temp_txs['block_time']).agg({'count'})
    temp_txs = temp_txs.reset_index()
    temp_corr['txs'] = temp_txs[('success','count')].to_list()
    
    #gas_price
    temp_corr['gas_price'] = create_gasprice_df(chain)['gas_price_gwei'].to_list()
    
    #unique addresses
    temp_uniqueaddresses = {}

    for d in dates:
        temp = chain[chain['block_time']==d]
        n_unique = temp['from'].nunique()
        temp_uniqueaddresses[d] = n_unique
        
    uniqueaddresses_temp = pd.DataFrame.from_dict({'block_time':temp_uniqueaddresses.keys(),
                                 'counts':temp_uniqueaddresses.values()})
    
    temp_corr['unique_addresses'] = uniqueaddresses_temp['counts'].to_list()
    
    #!!!number of contracts deployed - zkSync doesn't have chain data on Scroll officially
    
    return temp_corr
```

### Create Correlation Data Frame
```python
corr_zksync = create_corr_df(all_zksync)
```

### Correlation Data Frame
```python
corr_zksync
```

### Function to Plot Correlation between Transactions and Gas Price
```python
def plot_txs_gasprice_corr(corr_df,chain_name):
    #Corr. between txs and gas price
    plt.figure()
    sns.lmplot(x='txs',y='gas_price',data=corr_df,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
    plt.title(f"{chain_name}: Transactions vs. Gas Price (Gwei)")
    plt.xlabel('Transactions per day')
    plt.ylabel('Gas price (gwei)')
    corrcoef_temp = stats.pearsonr(corr_df['txs'], corr_df['gas_price'])
    plt.text(corr_df['txs'].iloc[corr_df['txs'].idxmax()], corr_df['gas_price'].iloc[corr_df['gas_price'].idxmax()], f"r={round(corrcoef_temp[0],2)}",ha='right', va='top')
```

### Plot Correlation between Transactions and Gas Price for zkSync Era
```python
plot_txs_gasprice_corr(corr_zksync,'zkSync Era')
```

```markdown
## Analysis of Protocol Distribution (Continued)

### Plot Top 20 Protocol Distribution for zkSync Era
```python
plot_topn_protocoldistribution(contracts_zksync,"zkSync Era",20)
```

### Plot Protocol Transaction Share for zkSync Era
```python
plot_protocolmarketshare(contracts_zksync,"zkSync Era",20)
```

### Plot Protocol Activity Over Time for zkSync Era
```python
plot_protocolactivity(contracts_zksync,"zkSync Era",20)
```

## Analysis of Protocol Distribution for Polygon

### Load Protocol Data for Polygon
```python
poly_protocols = pd.read_csv('/Users/cyrusleung/Desktop/Scroll/Chains Data/polygon txs/protocols/protocols_5-30_9-9.csv')
poly_protocols['block_time'] = poly_protocols['block_time'].map(lambda x: x.split()[0])
poly_protocols['block_time'] = pd.to_datetime(poly_protocols['block_time'],format='%Y-%m-%d')
```

### Sort and Clean Protocol Data for Polygon
```python
poly_protocols.sort_values(by='block_time',ascending=True,inplace=True)
poly_protocols = poly_protocols.drop_duplicates(subset=['hash'], keep='first')
```

### Plot Protocol Distribution for Polygon
```python
plot_protocoldistribution(poly_protocols,'Polygon')
```

### Plot Top 20 Protocol Distribution for Polygon
```python
plot_topn_protocoldistribution(poly_protocols,'Polygon',20)
```

### Plot Protocol Transaction Share for Polygon
```python
plot_protocolmarketshare(poly_protocols,'Polygon',20)
```

### Plot Protocol Activity Over Time for Polygon
```python
plot_protocolactivity(poly_protocols,'Polygon',20)
```

## Conclusion

This documentation covers an analysis of transaction data and protocol distribution for the Optimism, Arbitrum, zkSync Era, and Polygon chains. It includes correlation analysis between transaction data, visualization of protocol distribution, and insights into the percentage of contract deployments in transactions.

For any further questions or assistance, please don't hesitate to reach out.
```



