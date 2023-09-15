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

### Data Analysis

The code then performs data analysis on blockchain data from Arbitrum, Optimism, and Polygon. It loads and preprocesses data for each chain, generates various visualizations, and identifies the most popular contracts and events.

The documentation provided covers the structure and purpose of the code. To use this code, you'll need to have the required libraries installed and set up the necessary API keys and environment variables as needed. Additionally, you can customize the code for your specific data analysis tasks and adapt the functions as necessary.
