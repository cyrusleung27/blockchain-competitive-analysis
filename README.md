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

Certainly! Here's the documentation for your code in markdown format:

```markdown
# Acquisition: How Many Users Does Each Protocol Bring?

## `first_time_users_protocol(protocol_df)`

This function takes a dataframe containing blockchain protocol data and returns two dataframes:
- `first_time_users_per_protocol`: Contains the number of first-time users each protocol attracted to the chain.
- `first_time_users_per_protocol_pct`: Contains the percentage of total new users each protocol brought to the chain.

## `plot_first_time_users_protocol(df, df_pct, topn, chain_name)`

This function plots the number of first-time users each protocol attracted to the chain and their percentage of total new users.
- Inputs:
  - `df`: Dataframe containing the number of first-time users per protocol.
  - `df_pct`: Dataframe containing the percentage of total new users per protocol.
  - `topn`: Number of top protocols to include in the plot.
  - `chain_name`: Name of the blockchain chain.

## Arbitrum

### Number of First-Time Users

```python
first_time_users_per_protocol_arb, first_time_users_per_protocol_pct_arb = first_time_users_protocol(protocol_arb)
```

### User Lifespan Analysis

```python
users_lifespan_by_protocol_arb = avg_users_lifespan_by_protocol(protocols_arb)
```

### User Frequency Analysis

```python
plot_chain_user_freq_topn(protocol_arb, 15, 'Arbitrum')
```

## Optimism

### Number of First-Time Users

```python
first_time_users_per_protocol_op, first_time_users_per_protocol_pct_op = first_time_users_protocol(protocols_op)
```

### User Lifespan Analysis

```python
users_lifespan_by_protocol_op_df = avg_users_lifespan_by_protocol(protocols_op, topn=20)
```

### User Frequency Analysis

```python
plot_chain_user_freq_topn(protocols_op, 15, 'Optimism')
```

This documentation provides an overview of the functions and analyses conducted on user acquisition and retention for Arbitrum and Optimism blockchain chains.
```
