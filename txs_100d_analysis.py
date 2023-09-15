#!/usr/bin/env python
# coding: utf-8

# In[104]:


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
#matplotlib.rcParams['font.family'] = "sans-serif"
import os
import requests

from tqdm import tqdm

import plotly.io as pio
pio.renderers.default = 'notebook'

get_ipython().run_line_magic('matplotlib', 'inline')


# In[255]:


#Clean up timestamp column to just have year-month-day
def clean_timestamp_transpose(t):
    return t.split('T')[0]
def clean_timestamp_dune(time):
    return datetime.datetime.strptime(time.split()[0], '%Y-%m-%d')
def string_to_datetime(t):
    return datetime.datetime.strptime(t,'%Y-%m-%d')
def clean_raw_dune(df):
    df_copy = df.copy()
    #sort by date and time
    df_copy = df_copy.sort_values(by=['block_time'])

    #remove duplicate transactions
    df_copy = df_copy.drop_duplicates(subset=['tx_hash'], keep='first')

    #Remove time (H:M:S) from timestamp
    df_copy['block_time'] = df_copy['block_time'].map(clean_timestamp_dune)
    
    return df_copy

def agg_dfs(path):
    filenames = glob.glob(path+'/*.csv')

    dfs = []
    for f in filenames:
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs,ignore_index=True)
    
    return df

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


# In[86]:


#test dune api/


# In[91]:


import dotenv
import os
import pandas as pd

from dune_client.types import QueryParameter
from dune_client.client import DuneClient
from dune_client.query import Query

query = Query(
    name="Chain user activity past month",
    query_id=2986002,
    params=[
        QueryParameter.text_type(name="TextField", value="Word"),
        QueryParameter.number_type(name="NumberField", value=3.1415926535),
        QueryParameter.date_type(name="DateField", value="2022-05-04 00:00:00"),
        QueryParameter.enum_type(name="ListField", value="Option 1"),
    ],
)
print("Results available at", query.url())

dotenv.load_dotenv()
dune = DuneClient(os.environ["iHBaf43wKvraQGyzF1ruDsSqRxq6dfqm"])
pd = dune.refresh_into_dataframe(query)


# In[88]:


response = requests.get('https://api.dune.com/api/v1/query/2986002/results/csv?api_key=iHBaf43wKvraQGyzF1ruDsSqRxq6dfqm')


# In[89]:


response


# In[ ]:





# # Last month (8/2023)

# In[5]:


def plot_contractsdistribution(protocol_df, title):
    print(protocol_df['contract_name'].value_counts())
    print(protocol_df['event_name'].value_counts())
    
    #Contract name
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=protocol_df['block_time'],
                                 columns=protocol_df['contract_name'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    year_month = protocol_df['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
    year_month = [ts.to_pydatetime().strftime('%Y-%m') for ts in year_month]
    dates = list(protocol_df['block_time'].unique())
    plt.xticks(np.linspace(0,len(dates),len(year_month)), year_month,rotation=40)
    plt.ylabel("Contracts (by proportion)")
    plt.title(f"{title}: Distribution of Contracts Over Time")
    
    #Event name
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=protocol_df['block_time'],
                                 columns=protocol_df['event_name'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    plt.xticks(np.linspace(0,len(dates),len(year_month)), year_month,rotation=40)
    plt.ylabel("Contracts (by proportion)")
    plt.ylabel("Events (by proportion)")
    plt.title(f"{title}: Distribution of Events Over Time")


# In[6]:


def plot_protocolmarketshare(contracts_df, chain_name, topn):
    
    norm = [float(i)/sum(contracts_df['namespace'].value_counts()) for i in contracts_df['namespace'].value_counts()[:topn]]
    norm = list(map(lambda x: round(x*100,1),norm))
    norm_dict = {k: v for k,v in list(zip(contracts_df['namespace'].value_counts()[:topn].index,norm))}

    plt.figure(figsize=(15,10))
    plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values())
    plt.title(f'{chain_name}: Protocols Transaction Share')
    plt.xticks(rotation=90)
    plt.xlabel('Protocol')
    plt.ylabel('Txs Share (%)')
    plt.bar_label(plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values()))
    print(contracts_df['namespace'].value_counts()[:topn].to_string())
    
    
    norm = [float(i)/sum(contracts_df['contract_name'].value_counts()) for i in contracts_df['contract_name'].value_counts()[:topn]]
    norm = list(map(lambda x: round(x*100,1),norm))
    norm_dict = {k: v for k,v in list(zip(contracts_df['contract_name'].value_counts()[:topn].index,norm))}

    plt.figure(figsize=(15,10))
    plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values())
    plt.title(f'{chain_name}: Contracts Transaction Share')
    plt.xticks(rotation=90)
    plt.xlabel('Contract')
    plt.ylabel('Txs Share (%)')
    plt.bar_label(plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values()))
    print(contracts_df['contract_name'].value_counts()[:topn].to_string())
    
    norm = [float(i)/sum(contracts_df['event_name'].value_counts()) for i in contracts_df['event_name'].value_counts()[:topn]]
    norm = list(map(lambda x: round(x*100,1),norm))
    norm_dict = {k: v for k,v in list(zip(contracts_df['event_name'].value_counts()[:topn].index,norm))}

    plt.figure(figsize=(15,10))
    plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values())
    plt.title(f'{chain_name}: Events Transaction Share')
    plt.xticks(rotation=90)
    plt.xlabel('Event')
    plt.ylabel('Txs Share (%)')
    plt.bar_label(plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values()))
    print(contracts_df['event_name'].value_counts()[:topn].to_string())
    
def most_popular_contracts(contracts_df, topn):

    temp = contracts_df['contract_name'].value_counts()[:topn].reset_index()['contract_name']
    for i in range(len(temp)):
        print(temp.iloc[i])
    return temp

def most_popular_events(contracts_df, topn):

    temp = contracts_df['event_name'].value_counts()[:topn].reset_index()['event_name']
    for i in range(len(temp)):
        print(temp.iloc[i])
    return temp
    


# ## Arbitrum

# In[5]:


path = 'Chains Data/arbitrum txs/2023-8'


# In[6]:


aug_arb = pd.read_csv(path+'/aug_arb.csv')


# In[7]:


aug_arb


# In[8]:


aug_arb['block_time'] = aug_arb['block_time'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))


# In[9]:


aug_arb['block_time'].iloc[0]


# In[10]:


plot_contractsdistribution(aug_arb,'Arbitrum (Aug 2023)')


# In[26]:


plot_protocolmarketshare(aug_arb,'Arbitrum (Aug 2023)',50)


# In[25]:


plot_protocolmarketshare(aug_arb,'Arbitrum (Aug 2023)',10)


# In[121]:


most_pop_contracts_arb = most_popular_contracts(aug_arb,100)


# In[80]:


most_pop_events_arb = most_popular_events(aug_arb,50)


# In[81]:


for event in most_pop_events_arb:
    print(event, aug_arb[aug_arb['event_name']==event]['namespace'].value_counts().index)


# In[100]:


aug_arb[aug_arb['contract_name']=='LPStaking']['namespace'].value_counts().index


# In[119]:


aug_op[aug_op['contract_name']=='OVMFiatToken']['namespace'].value_counts().index


# In[81]:


aug_arb[aug_arb['contract_name'].isin(most_pop_contracts_arb)]


# ## Optimism

# In[ ]:





# In[14]:


path = 'Chains Data/optimism/2023-8'
aug_op = agg_dfs(path)


# In[15]:


aug_op = clean_raw_dune(aug_op)


# In[16]:


aug_op


# In[122]:


most_popular_contracts(aug_op,100)


# In[82]:


most_pop_events_op = most_popular_events(aug_op,50)


# In[85]:


for event in most_pop_events_op:
    print(event, str(aug_op[aug_op['event_name']==event]['namespace'].value_counts().index))


# In[27]:


plot_protocolmarketshare(aug_op,'Optimism (Aug 2023)',50)


# In[28]:


plot_protocolmarketshare(aug_op,'Optimism (Aug 2023)',10)


# In[18]:


#What are smart contracts that devs on both Arb and Op are working on?
aug_arb_contracts = list(aug_arb['contract_name'].value_counts()[:50].index)
aug_op_contracts = list(aug_op['contract_name'].value_counts()[:50].index)
aug_contracts_arb_op = set(aug_arb_contracts).intersection(set(aug_op_contracts))


# In[31]:


#What are smart contracts that devs on both Arb and Op are working on?
aug_arb_protocols = list(aug_arb['namespace'].value_counts()[:10].index)
aug_op_protocols = list(aug_op['namespace'].value_counts()[:10].index)
aug_protocols_arb_op_t10 = set(aug_arb_protocols).intersection(set(aug_op_protocols))


# In[33]:


set(aug_arb_protocols) ^ set(aug_op_protocols)


# In[32]:


aug_protocols_arb_op_t10


# In[19]:


aug_contracts_arb_op


# In[20]:


aug_arb_contracts_valuecount = pd.DataFrame({'contract_name':list(aug_arb['contract_name'].value_counts()[:50].index),
                                            'count':list(aug_arb['contract_name'].value_counts()[:50])})
aug_op_contracts_valuecount = pd.DataFrame({'contract_name':list(aug_op['contract_name'].value_counts()[:50].index),
                                            'count':list(aug_op['contract_name'].value_counts()[:50])})


# In[21]:


aug_arb_contracts_valuecount


# In[22]:


aug_arb_contracts_valuecount[aug_arb_contracts_valuecount['contract_name'].isin(aug_contracts_arb_op)]


# In[60]:


aug_op_contracts_valuecount[aug_op_contracts_valuecount['contract_name'].isin(aug_contracts_arb_op)]


# In[ ]:





# In[ ]:





# In[ ]:





# ## Polygon
# Launch date: 5/31/2020

# In[3]:


#Combine data from 5/30 to 9/8, a roughly 100 day period
#Polygon mainnet launched 5/31
'''temps = ['Chains Data/polygon txs/polygon_0530-0731.csv',
         'Chains Data/polygon txs/polygon_0730-0813.csv',
         'Chains Data/polygon txs/polygon_0813-0817.csv',
         'Chains Data/polygon txs/polygon_0820-0829.csv',
         'Chains Data/polygon txs/polygon_0828-0903.csv',
         'Chains Data/polygon txs/polygon_0905-0908.csv'
        ]'''
temp1 = pd.read_csv('Chains Data/polygon txs/polygon_0530-0731.csv')
temp2 = pd.read_csv('Chains Data/polygon txs/polygon_0730-0813.csv')
temp3 = pd.read_csv('Chains Data/polygon txs/polygon_0813-0817.csv')
temp4 = pd.read_csv('Chains Data/polygon txs/polygon_0816-0821.csv')
temp5 = pd.read_csv('Chains Data/polygon txs/polygon_0820-0829.csv')
temp6 = pd.read_csv('Chains Data/polygon txs/polygon_0828-0903.csv')
temp7 = pd.read_csv('Chains Data/polygon txs/polygon_0905-0908.csv')


#Read and preprocess polygon data
poly = pd.concat([temp1,temp2,temp3,temp4,temp5,temp6,temp7],ignore_index=True)

poly['timestamp'] = poly['timestamp'].map(clean_timestamp_transpose)

poly['timestamp'] = pd.to_datetime(poly['timestamp'], format="%Y-%m-%d")

def month_day(t):
    return f"{t.month}-{t.day}"

poly['date'] = poly['timestamp'].map(month_day)

poly = poly.drop_duplicates(subset=['transaction_hash'], keep='first')

poly.sort_values(by='timestamp', inplace = True)


# In[4]:


poly


# In[4]:

#Create a dataframe for Polygon blockchain transactions


txs = poly.groupby(poly['timestamp']).agg({'count'})


# In[5]:


poly_txs = pd.read_csv('./Chains Data/polygon txs/polygon_txs.csv')


# In[6]:


poly_txs['Date(UTC)'] = pd.to_datetime(poly_txs['Date(UTC)'],format="%m/%d/%Y")


# In[8]:


poly_txs100 = poly_txs[poly_txs['Date(UTC)']<datetime.datetime(2020,9,8)]


# In[9]:


poly_txs100.iloc[-1]


# In[10]:

#Plot Polygon transactions per day over time (time-series chart)

plt.figure(1)
plt.plot(poly_txs100['Date(UTC)'], poly_txs100['Value'])
plt.xticks(rotation=40)
plt.title('Polygon transactions per day')


# ## Arbitrum
# Launch Date: 8/31/2021

# In[8]:

#Read and preprocess and clean Arbitrum blockchain transaction data
path = 'Chains Data/arbitrum'
protocol_arb = agg_dfs(path)


# In[9]:


protocol_arb = clean_raw_dune(protocol_arb)


# In[10]:


protocol_arb


# In[7]:


arb = pd.read_csv('Chains Data/arbitrum txs/arbitrum_txs.csv')


# In[8]:


arb['Date(UTC)'] = pd.to_datetime(arb['Date(UTC)'], format="%m/%d/%Y").dt.strftime('%Y-%m-%d')


# In[9]:


arb['Date(UTC)'] = pd.to_datetime(arb['Date(UTC)'], format='%Y-%m-%d')


# In[10]:


#arb_final = arb[arb['Date(UTC)'] < datetime.datetime.strptime('2021-08-20', '%Y-%m-%d')+datetime.timedelta(days=100)]
arb_final = arb[(arb['Date(UTC)'] < datetime.datetime(2021,12,10)) & (arb['Date(UTC)'] > datetime.datetime(2021,8,30))]


# In[11]:


arb_final


# In[9]:


#arb_final = arb_final[arb_final['Date(UTC)'] > datetime.datetime.strptime('2021-08-30', '%Y-%m-%d')]


# In[86]:

#Read and clean data on protocols on Arbitrum blockchain
arb_protocol = pd.read_csv('Chains Data/arbitrum txs/protocols/all_protocols.csv')


# In[87]:


arb_protocol.columns


# In[91]:


arb_protocol['block_time'] = arb_protocol['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))


# In[92]:


arb_protocol['block_time'].iloc[0]


# In[ ]:





# In[89]:


arb_protocol_final = arb_protocol['protocol_name'].value_counts()


# In[18]:


'''path = 'Chains Data/arbitrum txs/arb_txs_full'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
all_arb = pd.concat(dfs, ignore_index=True)

#sort by date and time
all_arb = all_arb.sort_values(by=['timestamp'])

#remove duplicate transactions
all_arb = all_arb.drop_duplicates(subset=['transaction_hash'], keep='first')

#Remove time (H:M:S) from timestamp
all_arb['timestamp'] = all_arb['timestamp'].map(clean_timestamp_transpose)'''


# In[161]:


all_arb = pd.read_csv('Chains Data/arbitrum txs/all_arb.csv').drop('Unnamed: 0',axis=1)


# In[162]:


all_arb


# In[163]:


arb_protocol


# # zkSync era
# Launch date: Mar 24, 2023

# In[12]:


'''path = 'Chains Data/zksync/zksync_era'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
all_zksync = pd.concat(dfs, ignore_index=True)

def clean_timestamp_dune(time):
    return datetime.datetime.strptime(time.split()[0], '%Y-%m-%d')

#sort by date and time
all_zksync = all_zksync.sort_values(by=['block_time'])

#remove duplicate transactions
all_zksync = all_zksync.drop_duplicates(subset=['hash'], keep='first')

#Remove time (H:M:S) from timestamp
all_zksync['block_time'] = all_zksync['block_time'].map(clean_timestamp_dune)

all_zksync.to_csv(path+'all_zksync.csv',index=False)'''


# In[13]:

#Read and preprocess zkSync Era blockchain data
path = 'Chains Data/zksync/zksync_era'
all_zksync = pd.read_csv(path+'all_zksync.csv')


# In[14]:


all_zksync['block_time'] = all_zksync['block_time'].map(clean_timestamp_dune)


# In[15]:


all_zksync = all_zksync[all_zksync['block_time']<datetime.datetime(2023,7,3)]


# In[16]:


all_zksync


# In[17]:


all_zksync['block_time'].iloc[0]


# In[18]:


#Read and preprocess data on protocols on zkSync era blockchain
contracts_zksync = pd.read_csv(path+'/contracts.csv')


# In[19]:


contracts_zksync['block_time'] = contracts_zksync['block_date']
contracts_zksync = contracts_zksync.drop('block_date',axis=1)
contracts_zksync['block_time'] = contracts_zksync['block_time'].map(clean_timestamp_dune)
contracts_zksync = contracts_zksync[contracts_zksync['block_time']<datetime.datetime(2023,7,3)]
contracts_zksync.sort_values(by='block_time',ascending=True,inplace=True)
contracts_zksync = contracts_zksync.drop_duplicates(subset=['hash'], keep='first')


# In[20]:


contracts_zksync


# In[21]:


contracts_zksync['block_time'].iloc[0]


# # Optimism

# Launch date: 2021-12-16

# In[22]:


'''path = 'Chains Data/optimism/'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
all_op = pd.concat(dfs, ignore_index=True)

#sort by date and time
all_op = all_op.sort_values(by=['block_time'])

#remove duplicate transactions
all_op = all_op.drop_duplicates(subset=['hash'], keep='first')

#Remove time (H:M:S) from timestamp
all_op['block_time'] = all_op['block_time'].map(clean_timestamp_dune)

#all_op.to_csv(path+'all_op.csv',index=False)'''


# In[11]:

#Read and preprocess Optimism data
path = 'Chains Data/optimism/'
all_op = pd.read_csv(path+'all_op.csv')
all_op['block_time'] = all_op['block_time'].map(clean_timestamp_dune)


# In[12]:


all_op


# In[13]:


all_op.columns


# In[14]:


all_op['block_time'].iloc[0]


# In[15]:


#Read and preprocess data on contracts and protocols on Optimism blockchain
path = 'Chains Data/optimism/contracts'
contracts_op = pd.read_csv(path+'contracts_op.csv')
'''filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
contracts_op = pd.concat(dfs, ignore_index=True)

contracts_op['block_time'] = contracts_op['block_date']
contracts_op = contracts_op.drop('block_date',axis=1)
contracts_op['block_time'] = contracts_op['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

#sort by date and time
contracts_op = contracts_op.sort_values(by=['block_time'])

#remove duplicate transactions
contracts_op = contracts_op.drop_duplicates(subset=['tx_hash'], keep='first')

#contracts_op.to_csv(path+'contracts_op.csv',index=False)'''


# In[16]:


contracts_op


# In[17]:


contracts_op['block_time'] = contracts_op['block_time'].map(clean_timestamp_dune)
contracts_op.iloc[0]['block_time']


# In[18]:


contracts_op['namespace'].value_counts()[:20]


# # Acquisition: how many users does each protocol bring?

# In[32]:


def first_time_users_protocol(protocol_df):
    '''
    Input the dataframe containing the blockchain's protocols' data and return a dataframe containing how many first time users each protocol attracted
    to the chain
    '''
    first_times = protocol_df.drop_duplicates(subset=['from']) #get all users' first time (row)
    first_time_users_per_protocol = first_times['namespace'].value_counts().reset_index()
    
    first_time_users_per_protocol_pct = round(first_times['namespace'].value_counts()/sum(first_times['namespace'].value_counts())*100,1)
    first_time_users_per_protocol_pct = first_time_users_per_protocol_pct.reset_index()

    return first_time_users_per_protocol, first_time_users_per_protocol_pct

def plot_first_time_users_protocol(df, df_pct, topn, chain_name):
    '''
    Plot the number of first time users each protocol attracted to the chain
    '''
    #Plot not pct
    fig, ax = plt.subplots()

    ax.barh(df.iloc[:topn]['namespace'], df.iloc[:topn]['count'], align='center')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(f'Number of new users each protocol brought to {chain_name}')
    ax.set_ylabel('Protocol')
    ax.set_title(f'Number of new users each protocol brought to {chain_name} (Top {topn})');
    ax.grid(axis='x')
    
    #Plot pct
    fig, ax = plt.subplots()

    ax.barh(df_pct.iloc[:topn]['namespace'], df_pct.iloc[:topn]['count'], align='center')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(f'Percent of total new users each protocol brought to {chain_name} (%)')
    ax.set_ylabel('Protocol')
    ax.set_title(f'Percent of total new users each protocol brought to {chain_name} (Top {topn})')
    ax.bar_label(ax.barh(df_pct.iloc[:topn]['namespace'], df_pct.iloc[:topn]['count'],align='center'));
    ax.grid(axis='x')
    ax.set_xlim(0,40)
    


# ## Arbitrum

# In[20]:

#Number of first time users each protocol attracted to Arbitrum
first_time_users_per_protocol_arb, first_time_users_per_protocol_pct_arb = first_time_users_protocol(protocol_arb)


# In[33]:

#Plot those first time users
plot_first_time_users_protocol(first_time_users_per_protocol_arb, first_time_users_per_protocol_pct_arb, topn=20, chain_name='Arbitrum')


# In[22]:


'''sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots()

# Plot the total crashes
sns.set_color_codes("bright")
sns.barplot(x="count", y="namespace", data=first_time_users_per_protocol_pct_arb.iloc[:20],
            label="count", color="b")

# Add informative axis label
ax.set(xlim=(0, 100), ylabel="Protocol",
       xlabel="Percent of total new users each protocol brought to Arbitrum (Top 20)")
sns.despine(left=True, bottom=True)'''


# ## Optimism

# In[23]:


protocols_op = pd.merge(contracts_op.rename(columns={'tx_hash':'hash'}), all_op[['hash','from','to']], on='hash')


# In[24]:


protocols_op


# In[25]:


first_times = protocols_op.drop_duplicates(subset=['from']) #get all users' first time (row)

first_time_users_per_protocol = first_times['namespace'].value_counts().reset_index()

    


# In[26]:


first_time_users_per_protocol.iloc[20]


# In[132]:


#Plot
fig, ax = plt.subplots()

# Example data
ax.barh(first_time_users_per_protocol.iloc[:20]['namespace'], first_time_users_per_protocol.iloc[:20]['count'], align='center')
#ax.set_yticks(first_time_users_per_protocol['count'], labels=first_time_users_per_protocol['namespace'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of new users each protocol brought to Optimism')
ax.set_ylabel('Protocol')
ax.set_title('Number of new users each protocol brought to Optimism (Top 20)')


# In[143]:


first_time_users_per_protocol_pct = round(first_times['namespace'].value_counts()/sum(first_times['namespace'].value_counts())*100,1)
first_time_users_per_protocol_pct = first_time_users_per_protocol_pct.reset_index()


# In[144]:


first_time_users_per_protocol_pct


# In[147]:


#Plot
fig, ax = plt.subplots()

# Example data
ax.barh(first_time_users_per_protocol_pct.iloc[:20]['namespace'], first_time_users_per_protocol_pct.iloc[:20]['count'], align='center')
#ax.set_yticks(first_time_users_per_protocol['count'], labels=first_time_users_per_protocol['namespace'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Percent of total new users each protocol brought to Optimism')
ax.set_ylabel('Protocol')
ax.set_title('Percent of total new users each protocol brought to Optimism (Top 20)')
ax.bar_label(ax.barh(first_time_users_per_protocol_pct.iloc[:20]['namespace'], first_time_users_per_protocol_pct.iloc[:20]['count'], align='center'));


# In[27]:

#Number of first time users each protocol attracted to Arbitrum
first_time_users_per_protocol_op, first_time_users_per_protocol_pct_op = first_time_users_protocol(protocols_op)


# In[34]:

#Plot those first time users per protocol
plot_first_time_users_protocol(first_time_users_per_protocol_op, first_time_users_per_protocol_pct_op,20,'Optimism')


# # User Retention (average lifespan users)

# In[284]:


def calculate_lifespan(user, df):
    '''
    Calculate the lifespan of a user
    '''
    if len(df[df['from']==user]) <= 1:
        return 0
    return (df[df['from']==user]['block_time'].iloc[-1] - df[df['from']==user]['block_time'].iloc[0]).days

def users_lifespan_per_protocol(protocol, users_by_protocol, protocols_df, sample_size=385):
    '''
    Calculate the lifespan of each user attracted to the chain by each respective protocol
    '''
    #Operation speed: 0.24 seconds per user
    if len(users_by_protocol[protocol]) < sample_size:
        return [calculate_lifespan(x, protocols_df) for x in users_by_protocol[protocol]]
    reduced_pop = np.random.choice(users_by_protocol[protocol], size=sample_size, replace=False)
    return [calculate_lifespan(x, protocols_df) for x in reduced_pop]
    '''
    n_users = len(users_by_protocol[protocol])
    if n_users <= 1000:
        return [calculate_lifespan(x, protocols_df) for x in users_by_protocol[protocol]]
    else:
        reduced_pop = np.random.choice(users_by_protocol[protocol], size=sample_size, replace=False)
        return [calculate_lifespan(x, protocols_df) for x in reduced_pop]'''
    
def avg_users_lifespan_by_protocol(protocols_df, topn=20):
    '''
    Return the mean, median, and standard deviation of each protocols' users' lifespan
    '''
    #Get a dataframe of all users' first times
    first_times = protocols_df.drop_duplicates(subset=['from']) #get all users' first time (row)
    
    #Dictionary. key: protocol, value: the users each protocol brought on
    users_by_protocol = {}

    for protocol in first_times['namespace'].unique():
        users_by_protocol[protocol] = list(first_times[['namespace','from']][first_times[['namespace','from']]['namespace']==protocol]['from'])

    #Get the lifespan of each user brought on by each protocol
    first_time_users_per_protocol, first_time_users_per_protocol_pct = first_time_users_protocol(protocols_df)
    users_lifespan_by_protocol = {}

    for protocol in tqdm(list(first_time_users_per_protocol['namespace'])[:20]):
        users_lifespan_by_protocol[protocol] = users_lifespan_per_protocol(protocol, 
                                                                          users_by_protocol,
                                                                         protocols_df)
    
    #Get mean, median, and standard deviations of the user lifespans
    mean_users_lifespan_by_protocol = {}
    median_users_lifespan_by_protocol = {}
    std_users_lifespan_by_protocol = {}

    for k, v in users_lifespan_by_protocol.items():
        mean_users_lifespan_by_protocol[k] = np.mean(v)

    for k, v in users_lifespan_by_protocol.items():
        median_users_lifespan_by_protocol[k] = np.median(v)

    for k, v in users_lifespan_by_protocol.items():
        std_users_lifespan_by_protocol[k] = np.std(v)

    #Turn these data into a df
    users_lifespan_by_protocol_df = pd.DataFrame({
    'Protocol':list(mean_users_lifespan_by_protocol.keys()),
    'Mean user lifespan':list(mean_users_lifespan_by_protocol.values()),
    'Median user lifespan':list(median_users_lifespan_by_protocol.values()),
    'Std user lifespan':list(std_users_lifespan_by_protocol.values())
    })

    return users_lifespan_by_protocol_df
    
def plot_users_lifespan_by_protocol(df, chain_name):
    '''
    Plot each protocols' user lifespan, mean median and standard deviation
    '''
    temp = pd.melt(df, id_vars=['Protocol'],value_vars=['Mean user lifespan','Median user lifespan','Std user lifespan'])
    temp = temp.rename(columns={'variable':'User lifespan type'})
    
    plt.figure(figsize=(6,18))
    sns.set_theme(style="whitegrid", palette="bright")
    sns.catplot(x = 'value', y='Protocol', 
            hue = 'User lifespan type',data=temp, 
            kind='bar')
    plt.title(f'{chain_name}: Average user lifespan of users brought by each protocol')
    plt.xlabel('Number of days')
    
def plot_avg_users_lifespan_by_protocol(df, chain_name,ordered=False):
    '''
    Plot just the mean user lifespan
    '''
    if ordered:
        plt.figure()
        fig, ax = plt.subplots()
        
        df = df.sort_values(by=['Mean user lifespan'],ascending=False)
        df['Mean user lifespan'] = df['Mean user lifespan'].map(round)
        ax.barh(df['Protocol'], df['Mean user lifespan'], align='center')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(f'Avg number of days')
        ax.set_ylabel('Protocol')
        ax.set_title(f'{chain_name}: Average user lifespan of users brought by each protocol (ordered)')
        ax.bar_label(ax.barh(df['Protocol'], df['Mean user lifespan'], align='center'));
        ax.grid(axis='y')
    
    else:
        plt.figure()
        fig, ax = plt.subplots()

        df['Mean user lifespan'] = df['Mean user lifespan'].map(round)
        ax.barh(df['Protocol'], df['Mean user lifespan'], align='center')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(f'Avg number of days')
        ax.set_ylabel('Protocol')
        ax.set_title(f'{chain_name}: Average user lifespan of users brought by each protocol (ordered)')
        ax.bar_label(ax.barh(df['Protocol'], df['Mean user lifespan'], align='center'));
        ax.grid(axis='y')
        
        
        
def plot_avg_users_lifespan_by_protocol_sns(df, chain_name,ordered=False):
    '''
    Same as the function above but use seaborn library for a different style
    '''
    if ordered:
        plt.figure()
        sns.set_theme(style="whitegrid",palette="deep")
        ax = sns.catplot(x = 'Mean user lifespan', y='Protocol', data=df.sort_values(by=['Mean user lifespan'],ascending=False), 
                kind='bar')
        plt.title(f'{chain_name}: Average user lifespan of users brought by each protocol (ordered)')
        plt.xlabel('Mean number of days')
        ax.bar_label(ax.containers[0], fmt='%.1f')
    else:
        plt.figure()
        sns.set_theme(style="whitegrid",palette="deep")
        ax = sns.catplot(x = 'Mean user lifespan', y='Protocol', data=df, 
                kind='bar')
        plt.title(f'{chain_name}: Average user lifespan of users brought by each protocol (ordered)')
        plt.xlabel('Mean number of days')
        ax.bar_label(ax.containers[0], fmt='%.1f')
    
def chain_user_freq_t10(protocol_df):
    '''
    Number of one-time, two-time, etc. all the way to ten-time users on a chain
    '''
    vc = protocol_df['from'].value_counts(ascending=True)
    temp = {}
    for i in range(1,11):
        temp[i] = len(vc[vc==i])
    
    temp_df = pd.DataFrame({'N Time User':temp.keys(), 'Count':temp.values()})
    return temp_df

def chain_user_freq(protocol_df,topn):
    '''
    Number of one-time, two-time, etc. users on a chain, you can specify how many X-time users to calculate up to
    '''
    vc = protocol_df['from'].value_counts(ascending=True)
    temp = {}
    for i in range(1,topn+1):
        temp[i] = len(vc[vc==i])
    
    temp_df = pd.DataFrame({'N Time User':temp.keys(), 'Count':temp.values()})
    return temp_df

def chain_user_freq_pct(protocol_df,topn):
    '''
    Percentage of total users are one-time, two-time, three-time, etc. users on a chain
    '''
    vc = protocol_df['from'].value_counts(ascending=True)
    temp = {}
    for i in range(1,topn+1):
        temp[i] = round(len(vc[vc==i])/protocol_df['from'].nunique()*100, 1) #not div by this, div by the number of unique addresses
    
    temp_df = pd.DataFrame({'N Time User':temp.keys(), 'Percent of Total Users':temp.values()})
    return temp_df

def plot_chain_user_freq(pop_df,title):
    '''
    Plot number of one-time,... users
    '''
    plt.figure(figsize=(50,30))
    pop_df.plot.bar(x='N Time User',y='Count',rot=360,title=title+': Number of one-time, two-time,... users')
    plt.xlabel(f'N-time Users (1-{len(pop_df)})')
    plt.ylabel('Count')
    plt.grid(False)
    
def plot_chain_user_freq_topn(protocol_df,topn,title):
    '''
    Plot number of one-time,... users, specifiying X time users
    '''
    pop_df = chain_user_freq(protocol_df,topn)
    
    plt.figure(figsize=(50,30))
    pop_df.plot.bar(x='N Time User',y='Count',rot=360,title=title+': Number of one-time, two-time,... users')
    plt.xlabel(f'N-time Users (1-{len(pop_df)})')
    plt.ylabel('Count')
    plt.grid(False)
    
def plot_chain_user_freq_topn_pct(protocol_df,topn,title):
    '''
    Plot percentage of total users are one-time, etc. users, specifiying X time users
    '''
    pop_df = chain_user_freq_pct(protocol_df,topn)
    
    plt.figure(figsize=(60,30))
    pop_df.plot.bar(x='N Time User',y='Percent of Total Users',rot=360,title=title+': Number of one-time, two-time,... users',color='orange')
    plt.xlabel(f'N-time Users (1-{len(pop_df)})')
    plt.ylabel('Percent of total users who are one-time,... users (%)')
    addlabels(pop_df['N Time User'], pop_df['Percent of Total Users'])
    plt.grid(False)


# In[276]:


protocols_arb['from'].nunique()


# In[214]:


'''plt.figure()
        sns.set_theme(style="whitegrid",palette="deep")
        ax = sns.catplot(x = 'Mean user lifespan', y='Protocol', data=df.sort_values(by=['Mean user lifespan'],ascending=False), 
                kind='bar')
        plt.title(f'{chain_name}: Average user lifespan of users brought by each protocol (ordered)')
        plt.xlabel('Mean number of days')
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.1f')'''


# In[166]:


users_by_protocol_op


# In[171]:


first_time_users_per_protocol_arb.iloc[:20]


# In[92]:


erc1155_lifespan = users_lifespan_per_protocol('erc1155',users_by_protocol_op,protocols_op)


# In[61]:


calculate_lifespan(users_by_protocol_op['erc20'][0], protocols_op)


# ## Arbitrum

# In[169]:


protocols_arb = protocol_arb


# In[170]:

#Calculate and plot user lifespan per protocol for Arbitrum chain
users_lifespan_by_protocol_df = avg_users_lifespan_by_protocol(protocols_arb)


# In[172]:


users_lifespan_by_protocol_arb = users_lifespan_by_protocol_df


# In[173]:


plot_users_lifespan_by_protocol(users_lifespan_by_protocol_arb, 'Arbitrum')


# In[224]:


plot_avg_users_lifespan_by_protocol(users_lifespan_by_protocol_arb, 'Arbitrum')


# In[225]:


plot_avg_users_lifespan_by_protocol(users_lifespan_by_protocol_arb, 'Arbitrum',ordered=True)


# In[227]:


plot_avg_users_lifespan_by_protocol_sns(users_lifespan_by_protocol_arb, 'Arbitrum',ordered=True)


# In[233]:


chain_user_freq_arb_100 = chain_user_freq(protocols_arb, 100)


# In[234]:


chain_user_freq_arb_10 = chain_user_freq(protocols_arb, 10)


# In[269]:


plot_chain_user_freq_topn(protocols_arb,15,'Arbitrum')


# In[240]:


plot_chain_user_freq(chain_user_freq_arb_100,'Arbitrum')


# In[285]:


plot_chain_user_freq_topn_pct(protocols_arb,15,'Arbitrum')


# ## Optimism

# In[35]:

#Calculate and plot user lifespan per protocol for Optimism chain
first_times = protocols_op.drop_duplicates(subset=['from']) #get all users' first time (row)


# In[37]:


first_times[['namespace','from']]


# In[44]:


list(first_times[['namespace','from']][first_times[['namespace','from']]['namespace']=='erc20']['from'])


# In[45]:


users_by_protocol_op = {}

for protocol in first_times['namespace'].unique():
    users_by_protocol_op[protocol] = list(first_times[['namespace','from']][first_times[['namespace','from']]['namespace']==protocol]['from'])


# In[82]:


first_times[first_times['namespace']=='sablier_v1_1']


# In[63]:


len(users_by_protocol_op['erc20'])


# In[89]:


len(users_by_protocol_op['erc1155'])


# In[90]:


users_by_protocol_op['erc1155']


# In[88]:


list(users_by_protocol_op.keys())[:20]


# In[103]:


list(first_time_users_per_protocol_op['namespace'])[:20]


# In[105]:


users_lifespan_by_protocol_op = {}

for protocol in tqdm(list(first_time_users_per_protocol_op['namespace'])[:20]):
    users_lifespan_by_protocol_op[protocol] = users_lifespan_per_protocol(protocol, 
                                                                          users_by_protocol_op,
                                                                         protocols_op)


# In[120]:


users_lifespan_by_protocol_op['wbtc'] = wbtc_lifespan


# In[110]:


set(list(first_time_users_per_protocol_op['namespace'])[:20]) - set(list(users_lifespan_by_protocol_op.keys()))


# In[113]:


wbtc_lifespan = users_lifespan_per_protocol('wbtc',users_by_protocol_op,protocols_op)


# In[118]:


protocols_op[protocols_op['from']=='0x1bc4c23691b5f68a6b408571b34aff15d519a67d']['block_time'].iloc[-1] - protocols_op[protocols_op['from']=='0x1bc4c23691b5f68a6b408571b34aff15d519a67d']['block_time'].iloc[0]


# In[114]:


wbtc_lifespan


# In[125]:


np.median(wbtc_lifespan)


# In[130]:


mean_users_lifespan_by_protocol_op = {}
median_users_lifespan_by_protocol_op = {}
std_users_lifespan_by_protocol_op = {}

for k, v in users_lifespan_by_protocol_op.items():
    mean_users_lifespan_by_protocol_op[k] = np.mean(v)
    
for k, v in users_lifespan_by_protocol_op.items():
    median_users_lifespan_by_protocol_op[k] = np.median(v)
    
for k, v in users_lifespan_by_protocol_op.items():
    std_users_lifespan_by_protocol_op[k] = np.std(v)



# In[127]:


mean_users_lifespan_by_protocol_op


# In[128]:


median_users_lifespan_by_protocol_op


# In[131]:


std_users_lifespan_by_protocol_op


# In[133]:


users_lifespan_by_protocol_op_df = pd.DataFrame({
    'Protocol':list(mean_users_lifespan_by_protocol_op.keys()),
    'Mean user lifespan':list(mean_users_lifespan_by_protocol_op.values()),
    'Median user lifespan':list(median_users_lifespan_by_protocol_op.values()),
    'Std user lifespan':list(std_users_lifespan_by_protocol_op.values())
})


# In[134]:


users_lifespan_by_protocol_op_df


# In[137]:


temp_df = pd.melt(users_lifespan_by_protocol_op_df, id_vars=['Protocol'],value_vars=['Mean user lifespan','Median user lifespan','Std user lifespan'])


# In[140]:


temp_df = temp_df.rename(columns={'variable':'User lifespan type'})


# In[141]:


temp_df


# In[142]:


sns.catplot(x = 'value', y='Protocol', 
            hue = 'User lifespan type',data=temp_df, 
            kind='bar')


# In[157]:


plot_users_lifespan_by_protocol(users_lifespan_by_protocol_op_df, 'Optimism')


# In[228]:


plot_avg_users_lifespan_by_protocol(users_lifespan_by_protocol_op_df, 'Optimism')


# In[229]:


plot_avg_users_lifespan_by_protocol(users_lifespan_by_protocol_op_df, 'Optimism',ordered=True)


# In[164]:


plot_avg_users_lifespan_by_protocol_sns(users_lifespan_by_protocol_op_df, 'Optimism')


# In[186]:


plot_avg_users_lifespan_by_protocol(users_lifespan_by_protocol_op_df, 'Optimism',ordered=True)


# In[286]:


plot_chain_user_freq_topn_pct(protocols_op,15,'Optimism')


# # Protocol User Retention

# In[36]:


def userretention_mom(protocol_df,app='Dune'):
    '''
    Func: For an interval of every month, calculate the percentage of users that returned
    '''
    userretention_df = pd.DataFrame(columns=['Month','User Retention (%)'])
    
    if app=='Dune':
        year_month = protocol_df['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
        year_month = [ts.to_pydatetime() for ts in year_month]
        df_copy = protocol_df.copy()
        df_copy['Year'] = df_copy['block_time'].dt.year
        df_copy['Month'] = df_copy['block_time'].dt.month

        add = {} #addresses

        for ym in year_month:
            add[ym] = list(df_copy[(df_copy['Year']==ym.year) & (df_copy['Month']==ym.month)]['from'].unique())

        pct = {}
        dts = list(add.keys())
        vals = list(add.values())

        for i in range(1,len(add)):
            pct[dts[i].strftime('%Y-%m')] = round((len(set(vals[i-1]) & set(vals[i]))/len(vals[i-1]))*100,0)

        userretention_df['Month'] = list(pct.keys())
        userretention_df['User Retention (%)'] = list(pct.values())

        return userretention_df
    
    elif app=='Transpose':
        year_month = protocol_df['timestamp'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
        year_month = [ts.to_pydatetime() for ts in year_month]
        df_copy = protocol_df.copy()
        df_copy['Year'] = df_copy['timestamp'].dt.year
        df_copy['Month'] = df_copy['timestamp'].dt.month

        add = {} #addresses

        for ym in year_month:
            add[ym] = list(df_copy[(df_copy['Year']==ym.year) & (df_copy['Month']==ym.month)]['from_address'].unique())

        pct = {}
        dts = list(add.keys())
        vals = list(add.values())

        for i in range(1,len(add)):
            pct[dts[i].strftime('%Y-%m')] = round((len(set(vals[i-1]) & set(vals[i]))/len(vals[i-1]))*100,0)

        userretention_df['Month'] = list(pct.keys())
        userretention_df['User Retention (%)'] = list(pct.values())

        return userretention_df
        

def plot_userretention(ur_df,title,calendar_dts=True):
    
    plt.figure()
    if calendar_dts:
        plt.bar(ur_df['Month'],ur_df['User Retention (%)'])
        plt.xticks(rotation=40)
    else:
        ur_df_copy = ur_df.copy()
        ur_df_copy['Month'] = [str(i) for i in range(1,len(ur_df_copy)+1)]
        plt.bar(ur_df_copy['Month'],ur_df['User Retention (%)'])
    plt.xlabel('Month')
    plt.ylabel('User Retention MoM (%)')
    plt.title(f'{title}: User Retention MoM (%)')
    
    plt.figure()
    if calendar_dts:
        plt.plot(ur_df['Month'],ur_df['User Retention (%)'])
        plt.xticks(rotation=40)
    else:
        ur_df_copy = ur_df.copy()
        ur_df_copy['Month'] = [str(i) for i in range(1,len(ur_df_copy)+1)]
        plt.plot(ur_df_copy['Month'],ur_df_copy['User Retention (%)'])
    plt.xlabel('Month')
    plt.ylabel('User Retention MoM (%)')
    plt.title(f'{title}: User Retention MoM (%)')
    

#Calculate the new user retention rates month-over-month for Polygon, zkSync, Optimisma  and Arbitrum
# ## Polygon

# In[31]:


poly


# In[51]:


userretention_poly = userretention_mom(poly,app='Transpose')


# In[52]:


userretention_poly


# In[54]:


plot_userretention(userretention_poly,'Polygon')


# ## Arbitrum

# In[33]:


all_arb


# In[61]:


all_arb['timestamp'] = all_arb['timestamp'].map(string_to_datetime)


# In[62]:


all_arb['timestamp'].iloc[0]


# In[63]:


userretention_arb = userretention_mom(all_arb,app='Transpose')


# In[64]:


plot_userretention(userretention_arb,title='Arbitrum')


# ## zkSync

# In[65]:


userretention_zksync = userretention_mom(all_zksync)


# In[67]:


userretention_zksync


# In[66]:


plot_userretention(userretention_zksync,title='zkSync')


# In[76]:


all_zksync[all_zksync['block_time'].dt.month == 7]['from'].tolist() in all_zksync[all_zksync['block_time'].dt.month == 6]['from'].tolist()


# In[84]:


all_zksync[all_zksync['block_time'].dt.month == 7]['from']


# In[85]:


all_zksync[all_zksync['block_time'].dt.month == 6]['from']


# ## Optimism

# In[79]:


userretention_op = userretention_mom(all_op)


# In[80]:


userretention_op


# In[83]:


plot_userretention(userretention_op,title='Optimism')


# # Uniswap data on Optimism and Arbitrum
# 
# 
# Optimism

# In[29]:


#contracts
path = 'Chains Data/optimism/uniswap'
uniswap_op = pd.read_csv(path+'uniswap_op.csv')
uniswap_op['block_time'] = uniswap_op['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
'''filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
uniswap_op = pd.concat(dfs, ignore_index=True)

uniswap_op['block_time'] = uniswap_op['block_time'].map(lambda x: datetime.datetime.strptime(x.split()[0], '%Y-%m-%d'))

#sort by date and time
uniswap_op = uniswap_op.sort_values(by=['block_time'])

#remove duplicate transactions
uniswap_op = uniswap_op.drop_duplicates(subset=['tx_hash'], keep='first')

#uniswap_op.to_csv(path+'uniswap_op.csv',index=False)'''


# In[30]:


uniswap_op


# In[23]:


uniswap_op['block_time'] = pd.to_datetime(uniswap_op['block_time'])


# In[24]:


uniswap_op


# In[25]:


uniswap_op.iloc[0]['block_time']


# ### Activity

# In[57]:


def protocol_txs(protocol_df):
    temp_txs = protocol_df.groupby(protocol_df['block_time']).agg({'count'})
    temp_txs = temp_txs.reset_index()
    return temp_txs
    
def plot_protocoltxs(txs_df,title,launch_date):
    plt.figure()
    plt.plot(txs_df['block_time'],txs_df[('namespace','count')])
    plt.axvline(launch_date,c='red')
    plt.title(f'{title} Txs/d')
    plt.xticks(rotation=40)


# In[27]:


uniswap_op_txs = uniswap_op.groupby(uniswap_op['block_time']).agg({'count'})
uniswap_op_txs = uniswap_op_txs.reset_index()


# In[28]:


uniswap_op_txs


# In[29]:


plt.figure()
plt.plot(uniswap_op_txs['block_time'],uniswap_op_txs[('namespace','count')])
plt.axvline(datetime.datetime(2021, 12, 16),c='red')
plt.title('Uniswap (Optimism) Txs/d')
plt.xticks(rotation=40)


# ### Most Used Contracts

# In[30]:


type(list(uniswap_op['block_time'].unique())[0])


# In[31]:


'''dates = list(uniswap_op['block_time'].unique())
dates = [ts.to_pydatetime() for ts in dates]'''


# In[14]:


plt.figure()
cross_tab_prop = pd.crosstab(index=uniswap_op['block_time'],
                             columns=uniswap_op['contract_name'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10',
                    #legend=None,
                    figsize=(15, 10))

plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
plt.xlabel("Date")
plt.ylabel("Contracts (by proportion)")


# In[15]:


def separate_contracts_df(protocol_df):
    
    cols = np.append(['Date'],protocol_df['contract_name'].unique())
    dates = protocol_df['block_time'].unique()
    contract_cts = {k: [] for k in cols[1:]}
    
    for dt in dates:
        temp_df = protocol_df[protocol_df['block_time']==dt]
        cts = temp_df['contract_name'].value_counts()
        for col in cols[1:]:
            if col in cts.keys():
                contract_cts[col].append(cts[col])
            else:
                contract_cts[col].append(0)
            
    df_copy = pd.DataFrame(contract_cts)
    df_copy['Date'] = dates
            
    return df_copy
    
    
#def separate_events_df(protocol_df):
    
    


# In[16]:


sep_contract_op = separate_contracts_df(uniswap_op)


# In[17]:


sep_contract_op


# In[18]:


sep_contract_op.plot(x='Date', kind='bar', stacked=True,
        title='Stacked Bar Graph by dataframe')

plt.xticks(sep_contract_op['Date'],rotation=40)


# In[58]:


def plot_contractsdistribution(protocol_df, protocol_name, chain_name):
    '''
    Plot the market share of protocols on a chain over time
    '''
    print(protocol_df['contract_name'].value_counts())
    print(protocol_df['event_name'].value_counts())
    
    #Contract name
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=protocol_df['block_time'],
                                 columns=protocol_df['contract_name'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    year_month = protocol_df['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
    year_month = [ts.to_pydatetime().strftime('%Y-%m') for ts in year_month]
    dates = list(protocol_df['block_time'].unique())
    plt.xticks(np.linspace(0,len(dates),len(year_month)), year_month,rotation=40)
    plt.ylabel("Contracts (by proportion)")
    plt.title(f"{protocol_name} ({chain_name}): Distribution of Contracts Over Time")
    
    #Event name
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=protocol_df['block_time'],
                                 columns=protocol_df['event_name'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    plt.xticks(np.linspace(0,len(dates),len(year_month)), year_month,rotation=40)
    plt.ylabel("Contracts (by proportion)")
    plt.ylabel("Events (by proportion)")
    plt.title(f"{protocol_name} ({chain_name}): Distribution of Events Over Time")


# In[33]:


year_month = uniswap_op['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
year_month = [ts.to_pydatetime() for ts in year_month]


# In[35]:


len(year_month)


# In[42]:


len(uniswap_op)


# In[44]:


plot_contractsdistribution(uniswap_op,'Uniswap','Optimism')


# ### Uniswap Optimism unique addresses per day

# In[35]:


uniswap_op[uniswap_op['block_time']==dates[0]]


# In[59]:


def protocol_uniqueaddresses(df):
    '''
    Plot the number of users per day
    '''
    dates = list(df['block_time'].unique())
    nuniqueaddresses = pd.DataFrame(columns=['Date','Nuniqueaddresses'])
    nuniqueaddresses['Date'] = dates
    temp_nunique = []
    for date in dates:
        temp_nunique.append(df[df['block_time']==date]['from'].nunique())
    nuniqueaddresses['Nuniqueaddresses'] = temp_nunique
    return nuniqueaddresses


# In[60]:


def plot_uniqueaddresses(unique_df,title_name):
    plt.figure()
    plt.plot(unique_df['Date'],unique_df['Nuniqueaddresses'])
    plt.xlabel('Date')
    plt.xticks(rotation=40)
    plt.ylabel('Number of Unique Addresses')
    plt.title(f"{title_name}: Number of Unique Addresses Per Day")


# In[37]:


uniswap_op_uniqueaddresses = protocol_uniqueaddresses(uniswap_op)
uniswap_op_uniqueaddresses


# In[41]:


plot_uniqueaddresses(uniswap_op_uniqueaddresses,"Uniswap (Optimism)")


# ### User Retention

# In[51]:


year_month = uniswap_op['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
year_month = [ts.to_pydatetime() for ts in year_month]


# In[57]:


y = year_month[0].year
m = year_month[0].month


# In[54]:


uniswap_op['block_time'].dt.year


# In[59]:


df_copy = uniswap_op.copy()


# In[61]:


df_copy['Year'] = df_copy['block_time'].dt.year
df_copy['Month'] = df_copy['block_time'].dt.month
df_copy[(df_copy['Year']==y) & (df_copy['Month']==m)]


# In[65]:


type(df_copy[(df_copy['Year']==y) & (df_copy['Month']==m)]['from'].unique())


# In[176]:


year_month = uniswap_arb['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
year_month = [ts.to_pydatetime() for ts in year_month]


# In[177]:


year_month


# In[50]:


def userretention_mom(protocol_df):
    '''
    Func: For an interval of every month, calculate the percentage of users that returned
    '''
    userretention_df = pd.DataFrame(columns=['Month','User Retention (%)'])
    year_month = protocol_df['block_time'].dt.to_period('M').unique().map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m'))
    year_month = [ts.to_pydatetime() for ts in year_month]
    df_copy = protocol_df.copy()
    df_copy['Year'] = df_copy['block_time'].dt.year
    df_copy['Month'] = df_copy['block_time'].dt.month
    
    add = {} #addresses
    
    for ym in year_month:
        add[ym] = list(df_copy[(df_copy['Year']==ym.year) & (df_copy['Month']==ym.month)]['from'].unique())
        
    pct = {}
    dts = list(add.keys())
    vals = list(add.values())
        
    for i in range(1,len(add)):
        pct[dts[i].strftime('%Y-%m')] = round((len(set(vals[i-1]) & set(vals[i]))/len(vals[i-1]))*100,0)
        
    userretention_df['Month'] = list(pct.keys())
    userretention_df['User Retention (%)'] = list(pct.values())
    
    return userretention_df

def plot_userretention(ur_df,title,calendar_dts=True):
    '''
    Plot user retention month over month
    '''
    plt.figure()
    if calendar_dts:
        plt.bar(ur_df['Month'],ur_df['User Retention (%)'])
        plt.xticks(rotation=40)
    else:
        ur_df_copy = ur_df.copy()
        ur_df_copy['Month'] = [str(i) for i in range(1,len(ur_df_copy)+1)]
        plt.bar(ur_df_copy['Month'],ur_df['User Retention (%)'])
    plt.xlabel('Month')
    plt.ylabel('User Retention MoM (%)')
    plt.title(f'{title}: User Retention MoM (%)')
    
    plt.figure()
    if calendar_dts:
        plt.plot(ur_df['Month'],ur_df['User Retention (%)'])
        plt.xticks(rotation=40)
    else:
        ur_df_copy = ur_df.copy()
        ur_df_copy['Month'] = [str(i) for i in range(1,len(ur_df_copy)+1)]
        plt.plot(ur_df_copy['Month'],ur_df_copy['User Retention (%)'])
    plt.xlabel('Month')
    plt.ylabel('User Retention MoM (%)')
    plt.title(f'{title}: User Retention MoM (%)')
    


# In[51]:


userretention_pct = userretention_mom(uniswap_op)


# In[52]:


userretention_pct


# In[53]:


plot_userretention(userretention_pct,'Uniswap (Optimism)',calendar_dts=True)


# In[159]:


def protocol_popularity_t10(protocol_df):
    '''
    Number of one-time, two-timer, etc. till ten-time users on a protocol
    '''
    vc = protocol_df['from'].value_counts(ascending=True)
    temp = {}
    for i in range(1,11):
        temp[i] = len(vc[vc==i])
    
    temp_df = pd.DataFrame({'N Time User':temp.keys(), 'Count':temp.values()})
    return temp_df

def protocol_popularity(protocol_df,topn):
    '''
    Number of one-time, two-timer, etc. till ten-time users on a protocol, specifiying X-time users
    '''
    vc = protocol_df['from'].value_counts(ascending=True)
    temp = {}
    for i in range(1,topn+1):
        temp[i] = len(vc[vc==i])
    
    temp_df = pd.DataFrame({'N Time User':temp.keys(), 'Count':temp.values()})
    return temp_df

def plot_protocolpopularity(pop_df,title):
    '''
    Plot number of one-time, two-timer, etc. users on a protocol
    '''
    plt.figure(figsize=(50,30))
    pop_df.plot.bar(x='N Time User',y='Count',rot=360,title=title+': Number of one-time, two-time,... users')
    plt.xlabel(f'N-time Users (1-{len(pop_df)})')
    plt.ylabel('Count')


# In[156]:


protpopt10_op = protocol_popularity_t10(uniswap_op)

protpop_op = protocol_popularity(uniswap_op,100)


# In[157]:


protpopt10_op


# In[158]:


protpop_op


# In[160]:


plot_protocolpopularity(protpopt10_op,'Uniswap (Optimism)')


# In[161]:


plot_protocolpopularity(protpop_op,'Uniswap (Optimism)')


# Get Uniswap data deployed on Arbitrum and Optimism chains
# ## Arbitrum

# In[54]:


#contracts
path = 'Chains Data/arbitrum txs/uniswap'
uniswap_arb = pd.read_csv(path+'uniswap_arb.csv')
uniswap_arb['block_time'] = uniswap_arb['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
'''filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename)

# Concatenate all data into one DataFrame
uniswap_arb = pd.concat(dfs, ignore_index=True)

uniswap_arb['block_time'] = uniswap_arb['block_time'].map(lambda x: datetime.datetime.strptime(x.split()[0], '%Y-%m-%d'))

#sort by date and time
uniswap_arb = uniswap_arb.sort_values(by=['block_time'])

#remove duplicate transactions
uniswap_arb = uniswap_arb.drop_duplicates(subset=['tx_hash'], keep='first')

#uniswap_arb.to_csv(path+'uniswap_op.csv',index=False)'''


# In[55]:


uniswap_arb


# In[56]:


uniswap_arb['block_time'].iloc[0]


# ### Activity

# In[136]:


uniswap_txs_arb = protocol_txs(uniswap_arb)


# In[138]:


uniswap_txs_arb = uniswap_txs_arb.iloc[:-1]
uniswap_txs_arb


# In[139]:


plot_protocoltxs(uniswap_txs_arb,'Uniswap (Arbitrum)',datetime.datetime(2021,8,31))


# ### Most Used Contracts

# In[48]:


plot_contractsdistribution(uniswap_arb,'Uniswap','Arbitrum')


# ### Unique addresses

# In[181]:


uniswap_unique_arb = protocol_uniqueaddresses(uniswap_arb)


# In[142]:


uniswap_unique_arb


# In[143]:


plot_uniqueaddresses(uniswap_unique_arb,'Uniswap (Arbitrum)')


# ### User retention

# In[58]:


userretention_pct = userretention_mom(uniswap_arb)

#userretention_pct = userretention_pct.iloc[:-1]

userretention_pct


# In[59]:


plot_userretention(userretention_pct,'Uniswap (Arbitrum)',calendar_dts=True)


# ### Protocol Popularity

# In[162]:


protpopt10_arb = protocol_popularity_t10(uniswap_arb)


# In[163]:


protpop_arb = protocol_popularity(uniswap_arb,100)


# In[164]:


protpopt10_arb


# In[165]:


protpop_arb


# In[166]:


plot_protocolpopularity(protpopt10_arb,'Uniswap (Arbitrum)')


# In[168]:


plot_protocolpopularity(protpop_arb,'Uniswap (Arbitrum)')


# # Level Finance (on BNB)

# In[67]:

#Get the protocol Level finance's data and get its transactions per day, number of users per day, most popular events on the protocol

path = 'Chains Data/level_finance'


# In[68]:


level = agg_dfs(path)


# In[70]:


level = clean_raw_dune(level)


# In[72]:


level['block_time'].iloc[0]


# ## Transactions

# In[74]:


level_txs = protocol_txs(level)
plot_protocoltxs(level_txs,'Level Finance (BNB)', datetime.datetime(2022,12,26))


# ## Most Used Contracts

# In[75]:


plot_contractsdistribution(level,'Level Finance','BNB')


# ## Number of Unique Addresses per day

# In[76]:


level_unique = protocol_uniqueaddresses(level)


# In[77]:


plot_uniqueaddresses(level_unique,'Level Finance (BNB)')


# # GMX (on Arbitrum)

# In[35]:
#Get the protocol GMX's data and get its transactions per day, number of users per day, most popular events on the protocol

path = 'Chains Data/gmx'


# In[43]:


gmx = pd.read_csv(path+'/gmx.csv')


# In[49]:


gmx = gmx.drop(['value','gas_limit','effective_gas_price'],axis=1)


# In[50]:


gmx


# In[53]:


gmx['block_time'] = gmx['block_time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))


# In[55]:


gmx['block_time'].iloc[0]


# ## Transactions

# In[62]:


gmx_txs = protocol_txs(gmx)

plot_protocoltxs(gmx_txs,'GMX (Arbitrum)',datetime.datetime(2021,9,1))


# ## Most Used Contracts

# In[63]:


plot_contractsdistribution(gmx,'GMX','Arbitrum')


# ## Number of Unique Addresses per day

# In[64]:


gmx_unique = protocol_uniqueaddresses(gmx)


# In[66]:


plot_uniqueaddresses(gmx_unique,'GMX (Arbitrum)')


# # Get transactions per day chart for zkSync, Optimism, Arbitrum and Polygon

# In[96]:


zksync_txs = all_zksync.groupby(all_zksync['block_time']).agg({'count'})
#zksync_txs.index = zksync_txs.index.map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
zksync_txs = zksync_txs[zksync_txs.index<datetime.datetime(2023,7,3)]


# In[97]:


zksync_txs


# In[192]:


txs_op = all_op.groupby(all_op['block_time']).agg({'count'})


# In[194]:


txs_op.columns


# In[98]:


fig,ax = plt.subplots(2,2,figsize=(20,20))
ax1 = ax[0][0]
ax1.plot(poly_txs100['Date(UTC)'],poly_txs100['Value'])
ax1.axvline(datetime.datetime(2020, 5, 31),c='red')
ax1.set_title('Polygon Txs/d first 100d')

#Arbitrum
ax2 = ax[0][1]
ax2.plot(arb_final['Date(UTC)'],arb_final['Value'])
ax2.axvline(datetime.datetime(2021, 8, 31),c='red')
#ax2.axvline(datetime.datetime(2021, 10, 26),c='red')
ax2.set_title('Arbitrum Txs/d first 100d')


# In[92]:


plt.plot(poly_txs100['Date(UTC)'],poly_txs100['Value'])
plt.axvline(datetime.datetime(2020, 5, 31),c='red')
plt.title('Polygon Txs/d first 100d')
plt.xticks(rotation=40)


# In[195]:


plt.figure()
plt.plot(arb_final['Date(UTC)'],arb_final['Value'])
plt.axvline(datetime.datetime(2021, 8, 31),c='red')
#ax2.axvline(datetime.datetime(2021, 10, 26),c='red')
plt.title('Arbitrum Txs/d first 100d')
plt.xticks(rotation=40)


# In[196]:


zksync_txs.index[0]


# In[222]:


zksync_txs = zksync_txs.reset_index()
plt.figure()
plt.plot(zksync_txs['block_time'],zksync_txs[('success','count')])
plt.axvline(datetime.datetime(2023, 3, 24),c='red')
plt.axvline(datetime.datetime(2023, 3, 25),c='red')
plt.axvline(datetime.datetime(2023, 4, 2),c='red')
plt.axvline(datetime.datetime(2023, 4, 8),c='red')
plt.title('zkSync Era Txs/d first 100d')
plt.xticks(rotation=40)


# In[264]:


zksync_txs.head(15)


# In[204]:


txs_op.index = txs_op.index.map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))


# In[226]:


txs_op = txs_op.reset_index()
plt.figure()
plt.plot(txs_op['block_time'],txs_op[('success','count')])
plt.axvline(datetime.datetime(2021, 12, 16),c='red')
plt.title('Optimism Txs/d first 100d')
plt.xticks(rotation=40)


# In[237]:


txs100_poly = poly_txs100[(poly_txs100['Date(UTC)'] >= datetime.datetime(2020,5,31)) & (poly_txs100['Date(UTC)'] <= (datetime.datetime(2020,5,31)+datetime.timedelta(days=100)))]
txs100_arb = arb_final[(arb_final['Date(UTC)'] >= datetime.datetime(2021,8,31)) & (arb_final['Date(UTC)'] < (datetime.datetime(2021,8,31)+datetime.timedelta(days=100)))]
txs100_zksync = zksync_txs[(zksync_txs['block_time'] >= datetime.datetime(2023,3,24)) & (zksync_txs['block_time'] <= datetime.datetime(2023,3,24)+datetime.timedelta(days=100))]
txs100_op = txs_op[(txs_op['block_time'] >= datetime.datetime(2021,12,16)) & (txs_op['block_time'] < datetime.datetime(2021,12,16)+datetime.timedelta(days=100))]


# In[238]:


txs100_poly


# In[239]:


txs100_arb


# In[240]:


txs100_zksync


# In[241]:


txs100_op


# In[257]:


compare_txs = pd.DataFrame(columns=['Day','Polygon_txs','Arbitrum_txs','zkSync_txs','Optimism_txs'])
compare_txs['Day'] = list(range(0,100))
compare_txs['Polygon_txs'] = txs100_poly['Value'].to_list()
compare_txs['Arbitrum_txs'] = txs100_arb['Value'].to_list()
compare_txs['zkSync_txs'] = txs100_zksync[('success','count')]to_list()
compare_txs['Optimism_txs'] = txs100_op[('success','count')]to_list()


# In[258]:


plt.figure()
compare_txs.plot(x='Day',y=['Polygon_txs','Arbitrum_txs','zkSync_txs','Optimism_txs'],kind='line')
plt.xlabel('Days')
plt.ylabel('Txs')
plt.title('Txs first 100 days: Polygon vs. Arbitrum vs. zkSync vs. Optimism')
plt.yscale('log')


# In[259]:

#Compare the transactions per day of the 4 chains on the same figure
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

compare_txs_norm = pd.DataFrame(columns=['Day','Polygon_txs','Arbitrum_txs','zkSync_txs','Optimism_txs'])
compare_txs_norm['Day'] = list(range(0,100))
compare_txs_norm['Polygon_txs'] = normalize(txs100_poly['Value'].to_list())
compare_txs_norm['Arbitrum_txs'] = normalize(txs100_arb['Value'].to_list())
compare_txs_norm['zkSync_txs'] = normalize(txs100_zksync[('success','count')].to_list())
compare_txs_norm['Optimism_txs'] = normalize(txs100_op[('success','count')].to_list())


# In[260]:


plt.figure()
compare_txs_norm.plot(x='Day',y=['Polygon_txs','Arbitrum_txs','zkSync_txs','Optimism_txs'],kind='line')
plt.xlabel('Days')
plt.ylabel('Txs (normalized)')
plt.title('Txs first 100 days: Polygon vs. Arbitrum vs. zkSync vs. Optimism (normalized)')


# ## Analyze transactions

# In[261]:


#Analyze zkSync Era
zksync_txs.sort_values(by=[('success','count')],ascending=False).iloc[:10]


# In[ ]:


#Analyze polygon txs
poly_txs100.sort_values(by=['Value'],ascending=False).iloc[:10]


# In[28]:


txs[txs.index < '2020-7-23'].sort_values(by=txs.columns[0],ascending=False)


# In[29]:


arb_final.sort_values(by=['Value'],ascending=False).iloc[:10]


# # The number of smart contracts created

# In[30]:


# Arbitrum's most popular smart contracts
all_arb['contract_address'].isna().sum()/len(all_arb) #99.8% of txs are not smart contract creations


# In[31]:


all_arb[~all_arb['contract_address'].isna()]


# In[32]:


#!!!next: find frequencies of each unique smart contracts ->dist. graph
deploycontracts_arb = all_arb.copy()
deploycontractscount_arb = deploycontracts_arb.groupby(deploycontracts_arb['timestamp']).agg({'count'})
deploycontractscount_arb = deploycontractscount_arb.reset_index()


# In[33]:


deploycontractscount_arb['deployed_contracts_count'] = deploycontractscount_arb[('contract_address','count')]


# In[34]:


deploycontractscount_arb['timestamp'] = pd.to_datetime(deploycontractscount_arb['timestamp'], format="%Y-%m-%d")


# In[35]:


avgdeploycontractscount_arb = deploycontractscount_arb['deployed_contracts_count'].sum()/len(deploycontractscount_arb['deployed_contracts_count'])
stddeploycontractscount_arb = round(deploycontractscount_arb['deployed_contracts_count'].std())


# In[36]:


plt.figure(1)
plt.plot(deploycontractscount_arb['timestamp'],deploycontractscount_arb['deployed_contracts_count'])
plt.axvline(datetime.datetime(2021, 8, 31),c='red')
plt.title('Arbitrum: Number of Contracts Created Every Day')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=40)
plt.text(datetime.datetime(2021,8,31),230,'Arbitrum launch',rotation=90,c='green')
plt.text(datetime.datetime(2021,9,26),250,f"Avg. # of contracts created: {round(avgdeploycontractscount_arb)}+/-{stddeploycontractscount_arb} per day")


# In[37]:


#Polygon
deploycontracts_poly = poly.copy()
deploycontractscount_poly = deploycontracts_poly.groupby(deploycontracts_poly['timestamp']).agg({'count'})
deploycontractscount_poly = deploycontractscount_poly.reset_index()
deploycontractscount_poly['timestamp'] = pd.to_datetime(deploycontractscount_poly['timestamp'], format="%Y-%m-%d")
deploycontractscount_poly['deployed_contracts_count'] = deploycontractscount_poly[('contract_address','count')]


# In[38]:


avgdeploycontractscount_poly = deploycontractscount_poly['deployed_contracts_count'].sum()/len(deploycontractscount_poly['deployed_contracts_count'])
stddeploycontractscount_poly = round(deploycontractscount_poly['deployed_contracts_count'].std())


# In[39]:


plt.figure(2)
plt.plot(deploycontractscount_poly['timestamp'],deploycontractscount_poly['deployed_contracts_count'])
plt.axvline(datetime.datetime(2020, 5, 31),c='red')
plt.title('Polygon: Number of Contracts Created Every Day')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=40)
plt.text(datetime.datetime(2020,5,31),100,'Polygon launch',rotation=90,c='green')
plt.text(datetime.datetime(2020,6,4),200,f"Avg. # of contracts created: {round(avgdeploycontractscount_poly)}+/-{stddeploycontractscount_poly} per day")


# ## Below is the failed attempt. 
# 
# It only captures the prominent smart contract address calls. Per Peter's rec, I will instead look for the freq. of unique smart contract addresses in the settlement layer instead of the protocols layer (as many protocols don't register on the protocol layer).

# In[40]:


#Polygon Protocols
poly_protocols = pd.read_csv('Chains Data/polygon txs/protocols.csv')


# In[41]:


plt.figure(1,figsize=(20,20))
sns.countplot(x='protocol_name',data=poly_protocols)
plt.title('Polygon\'s Most Popular Protocols in the First 100d')
plt.xticks(rotation=40, ha="right")
plt.yticks([0,1])
plt.show()


# In[42]:


#Arbitrum Protocols
plt.figure(2,figsize=(20,20))
sns.countplot(x='protocol_name',data=arb_protocol)
plt.title('Arbitrum\'s Most Popular Protocols in the First 100d')
plt.xticks(rotation=40, ha="right")


# ## Unique addresses and their activity (unsure if this is correct)

# In[22]:


arb_protocols = arb_protocol.copy()


# In[23]:


arb_protocols.columns


# In[67]:


arb_protocols['from_address'].unique()


# In[68]:


arb_protocols['from_address'].value_counts()


# In[47]:


#Arbitrum Protocols address distribution
plt.figure(3,figsize=(20,20))
sns.countplot(x='from_address',data=arb_protocols)
plt.title('Arbitrum\'s Unique Protocol Address Activity Distribution')
plt.xticks(rotation=40, ha="right")


# ## Total gas used per day (Gwei) for the 4 blockchains

# In[48]:


poly_gas = poly.groupby(poly['timestamp']).agg({'sum'})
poly_gasprice = poly.groupby(poly['timestamp']).mean()


# In[49]:


poly_gas[('gas_used_gwei', 'sum')] = poly_gas[('gas_used', 'sum')]*poly_gasprice['gas_price']/1e9 #gas used convert to gas used in units to gas used in gwei


# In[50]:


arb_gas = all_arb[['timestamp','gas_price','gas_used']]


# In[51]:


#Not using group by function for the entire DataFrame because it takes too long to run
#Find average gas price per day
arb_avggasprice = arb_gas.groupby(arb_gas['timestamp']).mean()

#Find sum of gas used per day
arb_gasused = arb_gas.groupby(arb_gas['timestamp']).agg({'sum'})


# In[52]:


arb_avggasprice


# In[53]:


arb_gasused


# In[54]:


arb_gasused[('gas_used_gwei', 'sum')] = arb_gasused[('gas_used', 'sum')]*arb_avggasprice['gas_price']/1e9 #gas used convert to gas used in units to gas used in gwei


# In[55]:


arb_gasused.index = pd.to_datetime(arb_gasused.index)


# In[56]:


arb_gasused


# In[57]:


polygasused_min_max_scaled = poly_gas.copy()
column=('gas_used_gwei','sum')
polygasused_min_max_scaled[column] = (polygasused_min_max_scaled[column] - polygasused_min_max_scaled[column].min()) / (polygasused_min_max_scaled[column].max() - polygasused_min_max_scaled[column].min())


# In[58]:


arbgasused_min_max_scaled = arb_gasused.copy()
column=('gas_used_gwei','sum')
arbgasused_min_max_scaled[column] = (arbgasused_min_max_scaled[column] - arbgasused_min_max_scaled[column].min()) / (arbgasused_min_max_scaled[column].max() - arbgasused_min_max_scaled[column].min())


# In[59]:


arbgasused_min_max_scaled


# In[60]:


arbgasused_min_max_scaled = arbgasused_min_max_scaled.reset_index(drop=True)


# In[61]:


polygasused_min_max_scaled = polygasused_min_max_scaled.reset_index(drop=True)


# In[62]:


polygasused_min_max_scaled[column]


# In[63]:


#Plot total gas used (Arbitrum vs. Polygon) (normalized)
#Arbitrum launched: 8/31/2021 
#Polygon launched: 5/31/2020 
plt.figure(4)
plt.plot(polygasused_min_max_scaled[('gas_used_gwei','sum')])
plt.plot(arbgasused_min_max_scaled[('gas_used_gwei','sum')])
plt.xlabel('Days')
plt.ylabel('Gwei (normalized)')
plt.title('Polygon vs. Arbitrum: Total Gas Used/d first 100 days after mainnet launch (normalized)') 
plt.axvline(1,c='red') 
plt.axvline(12,c='purple') 
plt.text(1,0.2,'Polygon launch',rotation=90) 
plt.text(12,0.2,'Arbitrum launch',rotation=90)


# In[64]:


plt.figure(5)
plt.title('Polygon: Total Gas Used/d first 100 days after mainnet launch') 
plt.plot(poly_gas.index,poly_gas[('gas_used_gwei', 'sum')])
plt.axvline(datetime.datetime(2020, 5, 31),c='red')
plt.text(datetime.datetime(2020, 5, 31),0,'Polygon launch',rotation=90)
plt.xlabel('Date')
plt.ylabel('Gwei')
plt.xticks(rotation=45)


# In[65]:


plt.figure(6)
plt.title('Arbitrum: Total Gas Used/d first 100 days after mainnet launch') 
plt.plot(arb_gasused.index,arb_gasused[('gas_used_gwei', 'sum')])
plt.axvline(datetime.datetime(2021, 8, 31),c='red')
plt.text(datetime.datetime(2021, 8, 31),1e12,'Arbitrum launch',rotation=90)
plt.xlabel('Date')
plt.ylabel('Gwei')
plt.xticks(rotation=45)


# ## Gas price every day (Gwei)

# In[84]:


def create_gasprice_df(chain):
    try:
        temp_gasprice = chain.groupby(chain['timestamp']).mean()
    except:
        temp_gasprice = chain.groupby(chain['block_time']).mean()
    temp_gasprice['gas_price_gwei'] = temp_gasprice['gas_price'].map(lambda x: x/1e9) #convert wei to gwei per gas unit
    temp_gasprice = temp_gasprice.reset_index()
    #temp_gasprice['block_time'] = temp_gasprice['block_time'].map(clean_timestamp_dune)
    return temp_gasprice
    
def plot_gasprice(gasprice_df,chain_name,launch_date):
    plt.figure()
    plt.plot(gasprice_df['block_time'],gasprice_df['gas_price_gwei'])
    plt.title(f"{chain_name}: Gas Price first 100 days after mainnet launch")
    plt.axvline(launch_date,c='red')
    plt.ylabel('Gwei')
    plt.xlabel('Date')
    plt.xticks(rotation=40)
    
    
    


# In[66]:


#Average gas price per day
poly_gasprice = poly.groupby(poly['timestamp']).mean()
poly_gasprice['gas_price_gwei'] = poly_gasprice['gas_price'].map(lambda x: x/1e9) #convert wei to gwei per gas unit


# In[67]:


max(poly_gasprice['gas_price_gwei'])


# In[68]:


plt.figure(7)
poly_gasprice.plot(y=('gas_price_gwei'),legend=None)
plt.title('Polygon: Gas Price first 100 days after mainnet launch')
plt.axvline(datetime.datetime(2020, 5, 31),c='red')
plt.ylabel('Gwei')


# In[69]:


#Average gas price per day
arb_gasprice = all_arb[['gas_limit','gas_used','gas_price']].groupby(all_arb['timestamp']).mean()
arb_gasprice['gas_price_gwei'] = arb_gasprice['gas_price'].map(lambda x: x/1e9) #convert wei to gwei per gas unit


# In[70]:


max(arb_gasprice['gas_price_gwei'])


# In[71]:


arb_gasprice = arb_gasprice.reset_index()
arb_gasprice


# In[72]:


arb_gasprice['timestamp'] = pd.to_datetime(arb_gasprice['timestamp'], format="%Y-%m-%d")


# In[73]:


plt.figure(8)
plt.plot(arb_gasprice['timestamp'],arb_gasprice['gas_price_gwei'])
plt.title('Arbitrum: Gas Price first 100 days after mainnet launch')
plt.axvline(datetime.datetime(2021, 8, 31),c='red')
plt.xlabel('Date')
plt.ylabel('Gwei')
plt.xticks(rotation=40)


# In[81]:


'''zksync_gasprice = all_zksync.groupby(all_zksync['block_time']).mean()
#zksync_gasprice.index = zksync_gasprice.index.map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
zksync_gasprice = zksync_gasprice[zksync_gasprice.index<datetime.datetime(2023,7,3)]
zksync_gasprice['gas_price_gwei'] = zksync_gasprice['gas_price'].map(lambda x: x/1e9)'''


# In[82]:


'''plt.figure()
plt.plot(zksync_gasprice.index,zksync_gasprice['gas_price_gwei'])
plt.title('zkSync Era: Gas Price first 100 days after mainnet launch')
plt.axvline(datetime.datetime(2023, 3, 24),c='red')
plt.ylabel('Gwei')
plt.xticks(rotation=40)'''


# In[85]:


zksync_gasprice = create_gasprice_df(all_zksync)


# In[87]:


plot_gasprice(zksync_gasprice,'zkSync Era',datetime.datetime(2023,3,24))


# In[273]:


gasprice_op = create_gasprice_df(all_op)
gasprice_op


# In[274]:


plot_gasprice(gasprice_op,'Optimism',datetime.datetime(2021,12,16))


# In[285]:


gasprice_op.sort_values(by='gas_price_gwei',ascending=False)


# ## Unique Addresses per day aka number of users per day for the 4 blockchains

# In[76]:


'''path = 'Chains Data/arbitrum txs/arb_txs_full'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
all_arb = pd.concat(dfs, ignore_index=True)

#sort by date and time
all_arb = all_arb.sort_values(by=['timestamp'])

#remove duplicate transactions
all_arb = all_arb.drop_duplicates(subset=['transaction_hash'], keep='first')

all_arb.iloc[0]['timestamp'].split('T')[0]

def create_date_column(t):
    return t.split('T')[0]

all_arb['date'] = all_arb['timestamp'].map(create_date_column)'''


# In[77]:


#Count number of transactions per unique from_address
dates = np.unique([d for d in all_arb['timestamp']])


# In[78]:


'''unique_addresses = {}

for d in dates:
    temp = all_arb[all_arb['timestamp']==d]
    n_unique = temp['from_address'].nunique()
    unique_addresses[d] = n_unique
    
df_temp = pd.DataFrame.from_dict({'dates':unique_addresses.keys(),
                                 'counts':unique_addresses.values()})

df_temp.to_csv('Chains Data/arbitrum txs/unique_addresses.csv')'''
uniqueaddresses_arb = pd.read_csv('Chains Data/arbitrum txs/unique_addresses.csv').drop('Unnamed: 0',axis=1)


# In[79]:


dates = list(map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),uniqueaddresses_arb['dates']))


# In[136]:


#Plot
plt.figure(6,figsize=(20,10))
# data  where the index is the date
fig = px.line(uniqueaddresses_arb, x=uniqueaddresses_arb['dates'], y=uniqueaddresses_arb['counts'],
             labels={
                 'dates': 'Date',
                 'counts': 'Number of unique addresses each day'
             }, title='Arbitrum - Unique Addresses/d')

# Show plot 
fig.update_layout(xaxis_rangeslider_visible=True)
fig.add_vline(x=datetime.datetime(2021, 8, 31),line_width=3, line_dash="solid", line_color="red")
fig.show()

'''ax.plot(dates, list(unique_addresses.values()))
plt.axvline(datetime.datetime(2021, 8, 31),c='red')
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,12)))
plt.setp(ax.get_xticklabels(), rotation=0)'''


# Now to do the same for Polygon data:

# In[137]:


def year_month_day(t):
    return f"{t.year}-{t.month}-{t.day}"

poly['date'] = poly['timestamp'].map(year_month_day)


# In[138]:


#Count number of transactions per unique from_address
dates = np.unique([d for d in poly['timestamp']])


# In[139]:


poly_uniqueaddresses = {}

for d in dates:
    temp = poly[poly['timestamp']==d]
    n_unique = temp['from_address'].nunique()
    poly_uniqueaddresses[d] = n_unique


# In[140]:


poly_uniqueaddresses


# In[141]:


#Plot
plt.figure(7,figsize=(20,10))
# data  where the index is the date
uniqueaddresses_poly = pd.DataFrame.from_dict({'dates':poly_uniqueaddresses.keys(),
                                 'counts':poly_uniqueaddresses.values()})


# In[142]:


fig = px.line(uniqueaddresses_poly, x=uniqueaddresses_poly['dates'], y=uniqueaddresses_poly['counts'],
             labels={
                 'dates': 'Date',
                 'counts': 'Number of unique addresses each day'
             }, title='Polygon - Unique Addresses/d')

# Show plot 
fig.update_layout(xaxis_rangeslider_visible=True)
fig.add_vline(x=datetime.datetime(2020, 5, 31),line_width=3, line_dash="solid", line_color="red")
fig.show()


# In[153]:


zksync_uniqueaddresses = {}
dates = list(map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),all_zksync['block_time']))
dates = sorted(list(set(dates)))


# In[154]:


dates


# In[173]:


for d in dates:
    temp = all_zksync[all_zksync['block_time']==d]
    n_unique = temp['from'].nunique()
    zksync_uniqueaddresses[d] = n_unique


# In[174]:


all_zksync[all_zksync['block_time']==datetime.datetime(2023, 4, 1, 0, 0)]


# In[175]:


zksync_uniqueaddresses


# In[176]:


uniqueaddresses_zksync = pd.DataFrame.from_dict({'dates':zksync_uniqueaddresses.keys(),
                                 'counts':zksync_uniqueaddresses.values()})


# In[178]:


fig = px.line(uniqueaddresses_zksync, x=uniqueaddresses_zksync['dates'], y=uniqueaddresses_zksync['counts'],
             labels={
                 'dates': 'Date',
                 'counts': 'Number of unique addresses each day'
             }, title='zkSync Era - Unique Addresses/d')

# Show plot 
fig.update_layout(xaxis_rangeslider_visible=True)
fig.add_vline(x=datetime.datetime(2023, 3, 24),line_width=3, line_dash="solid", line_color="red")
fig.show()


# ## Address Activity Distribution

# # Is there a correlation between gas price and transactions per day, ...?

# In[88]:


zksync_txs


# In[ ]:





# In[147]:


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
    
    


# In[148]:


corr_zksync = create_corr_df(all_zksync)


# In[149]:


corr_zksync


# In[281]:


def plot_txs_gasprice_corr(corr_df,chain_name):
    #Corr. between txs and gas price
    plt.figure()
    sns.lmplot(x='txs',y='gas_price',data=corr_df,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
    plt.title(f"{chain_name}: Transactions vs. Gas Price (Gwei)")
    plt.xlabel('Transactions per day')
    plt.ylabel('Gas price (gwei)')
    corrcoef_temp = stats.pearsonr(corr_df['txs'], corr_df['gas_price'])
    plt.text(corr_df['txs'].iloc[corr_df['txs'].idxmax()], corr_df['gas_price'].iloc[corr_df['gas_price'].idxmax()], f"r={round(corrcoef_temp[0],2)}",ha='right', va='top')
    


# In[279]:


plot_txs_gasprice_corr(corr_zksync,'zkSync Era')


# In[275]:


corr_op = create_corr_df(all_op)


# In[286]:


corr_op.sort_values(by='gas_price',ascending=False)


# In[287]:


corr_op_woanom = corr_op[corr_op['block_time'] != datetime.datetime(2022,3,11)]


# In[282]:


plot_txs_gasprice_corr(corr_op,'Optimism')


# In[288]:


plot_txs_gasprice_corr(corr_op_woanom,'Optimism')


# In[77]:


poly_gasprice = poly_gasprice.reset_index() #in Gwei


# In[78]:


#Polygon
corr_poly = pd.DataFrame(columns=['timestamp','txs','gas_price','unique_addresses','contracts_deployed','total_gas_used'])
corr_poly['timestamp'] = poly_gasprice['timestamp']
corr_poly['timestamp'] = pd.to_datetime(corr_poly['timestamp'],format='%Y-%m-%d')
corr_poly = corr_poly.iloc[1:]

#txs
txs_poly = poly_txs.copy()
txs_poly = txs_poly.rename(columns={'Date(UTC)':'timestamp'})
corr_poly['txs'] = corr_poly.merge(txs_poly, how='inner', on='timestamp')['Value'].to_list()

#gas price
corr_poly['gas_price'] = poly_gasprice['gas_price_gwei']

#unique addresses
uniqueaddresses_poly = uniqueaddresses_poly.rename(columns={'dates':'timestamp'})
corr_poly['unique_addresses'] = corr_poly.merge(uniqueaddresses_poly, how='inner', on='timestamp')['counts'].to_list()

#number of contracts deployed
deploycontractscount_poly_copy = deploycontractscount_poly[['timestamp','deployed_contracts_count']]
deploycontractscount_poly_copy.columns = deploycontractscount_poly_copy.columns.droplevel(1)
corr_poly['contracts_deployed'] = corr_poly.merge(deploycontractscount_poly_copy, how='inner', on='timestamp')['deployed_contracts_count'].to_list()

#gas used
poly_gasused_copy = poly_gas.copy()
poly_gasused_copy = poly_gasused_copy.reset_index()
poly_gasused_copy['timestamp'] = pd.to_datetime(poly_gasused_copy['timestamp'],format='%Y-%m-%d')
poly_gasused_copy.columns = poly_gasused_copy.columns.droplevel(1)
corr_poly['total_gas_used'] = corr_poly.merge(poly_gasused_copy, how='inner', on='timestamp')['gas_used_gwei'].to_list()


# In[79]:


corr_poly


# In[80]:


#Corr. between txs and gas price
plt.figure(8)
sns.lmplot(x='txs',y='gas_price',data=corr_poly,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Polygon: Transactions vs. Gas Price (Gwei)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas price (gwei)')
corrcoef_poly = stats.pearsonr(corr_poly['txs'], corr_poly['gas_price'])
plt.text(5000,30,f"r={round(corrcoef_poly[0],2)}")


# In[81]:


#Corr. between txs and gas price
plt.figure()
sns.lmplot(x='txs',y='total_gas_used',data=corr_poly,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Polygon: Transactions vs. Total Gas Used (Gwei)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas Used (gwei)')
corrcoef_poly = stats.pearsonr(corr_poly['txs'], corr_poly['total_gas_used'])
plt.text(1000,1.5e10,f"r={round(corrcoef_poly[0],2)}")


# In[82]:


#Corr. between txs and gas price
plt.figure()
sns.lmplot(x='txs',y='total_gas_used',data=corr_poly,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Polygon: Transactions vs. Total Gas Used (Gwei)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas Used (gwei)')
corrcoef_poly = stats.pearsonr(corr_poly['txs'], corr_poly['total_gas_used'])
plt.text(1000,1.5e10,f"r={round(corrcoef_poly[0],2)}")


# In[83]:


corr_poly['txs']-corr_poly['contracts_deployed']


# In[84]:


corr_poly['txs']


# # What percent of transactions are contract deployments?

# In[ ]:





# In[85]:


#What percent of transactions are contract deployments?
plt.figure()
plt.plot(corr_poly['timestamp'],round(corr_poly['contracts_deployed']/corr_poly['txs'],3))
plt.title('Polygon: % of txs that are contract deployments')
plt.xlabel('Date')
plt.ylabel('Percent')
plt.xticks(rotation=40)
avg_percenttxscontracts_poly = round((corr_poly['contracts_deployed']/corr_poly['txs']).mean(),3)
plt.text(datetime.datetime(2020,8,15),0.35,f"Mean: {avg_percenttxscontracts_poly}")


# In[86]:


plt.figure()
cormat_poly = corr_poly.corr()
round(cormat_poly,2)
sns.heatmap(cormat_poly,annot=True)
plt.title('Polygon: Correlation Heatmap')


# In[87]:


cormat_poly


# In[88]:


#Arbitrum
corr_arb = pd.DataFrame(columns=['timestamp','txs','gas_price','unique_addresses','contracts_deployed', 'total_gas_used'])
corr_arb['timestamp'] = arb_gasprice['timestamp']
corr_arb['timestamp'] = pd.to_datetime(corr_arb['timestamp'],format='%Y-%m-%d')

txs_arb = arb.copy()
txs_arb = txs_arb.rename(columns={'Date(UTC)':'timestamp'})
corr_arb['txs'] = corr_arb.merge(txs_arb, how='inner', on='timestamp')['Value'].to_list()

corr_arb['gas_price'] = arb_gasprice['gas_price_gwei']

uniqueaddresses_arb = uniqueaddresses_arb.rename(columns={'dates':'timestamp'})
uniqueaddresses_arb['timestamp'] = pd.to_datetime(uniqueaddresses_arb['timestamp'],format='%Y-%m-%d')
corr_arb['unique_addresses'] = corr_arb.merge(uniqueaddresses_arb, how='inner', on='timestamp')['counts'].to_list()

deploycontractscount_arb_copy = deploycontractscount_arb[['timestamp','deployed_contracts_count']]
deploycontractscount_arb_copy.columns = deploycontractscount_arb_copy.columns.droplevel(1)
corr_arb['contracts_deployed'] = corr_arb.merge(deploycontractscount_arb_copy, how='inner', on='timestamp')['deployed_contracts_count'].to_list()

#gas used
arb_gasused_copy = arb_gasused.copy()
arb_gasused_copy = arb_gasused_copy.reset_index()
arb_gasused_copy['timestamp'] = pd.to_datetime(arb_gasused_copy['timestamp'],format='%Y-%m-%d')
arb_gasused_copy.columns = arb_gasused_copy.columns.droplevel(1)
corr_arb['total_gas_used'] = corr_arb.merge(arb_gasused_copy, how='inner', on='timestamp')['gas_used_gwei'].to_list()


# In[89]:


corr_arb[corr_arb['total_gas_used']==corr_arb['total_gas_used'].max()]


# In[90]:


#Corr. between txs and gas price
plt.figure(9)
sns.lmplot(x='txs',y='gas_price',data=corr_arb,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Arbitrum: Transactions vs. Gas Price (Gwei)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas price (gwei)')
corrcoef_arb = stats.pearsonr(corr_arb['txs'], corr_arb['gas_price'])
plt.text(25000,6,f"r={round(corrcoef_arb[0],2)}")


# In[91]:


#What is that anomaly?
print(max(corr_arb['gas_price']))
print(corr_arb[corr_arb['gas_price']==7.295112152469643])


# In[92]:


#Corr. between txs and gas price
plt.figure()
sns.lmplot(x='txs',y='total_gas_used',data=corr_arb,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Arbitrum: Transactions vs. Total Gas Used (Gwei)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas Used (gwei)')
corrcoef_arb = stats.pearsonr(corr_arb['txs'], corr_arb['total_gas_used'])
plt.text(25000,1e12,f"r={round(corrcoef_arb[0],2)}")


# In[93]:


corr_arb_withoutanomaly = corr_arb.loc[corr_arb['total_gas_used']!=corr_arb['total_gas_used'].max()]


# In[94]:


#Corr. between txs and gas price (removing the 9/12/23)
plt.figure()
sns.lmplot(x='txs',y='total_gas_used',data=corr_arb_withoutanomaly,scatter_kws={'color':'blue'},line_kws={'color': 'red'})
plt.title('Arbitrum: Transactions vs. Total Gas Used (Gwei) (without 9/12/23)')
plt.xlabel('Transactions per day')
plt.ylabel('Gas Used (gwei)')
corrcoef_arb_woanomaly = stats.pearsonr(corr_arb_withoutanomaly['txs'], corr_arb_withoutanomaly['total_gas_used'])
plt.text(25000,1.4e11,f"r={round(corrcoef_arb_woanomaly[0],2)}")


# In[95]:


#What percent of transactions are contract deployments?
plt.figure()
plt.plot(corr_arb['timestamp'],round(corr_arb['contracts_deployed']/corr_arb['txs'],3))
plt.title('Arbitrum: % of txs that are contract deployments')
plt.xlabel('Date')
plt.ylabel('Percent')
plt.xticks(rotation=40)
avg_percenttxscontracts_arb = round((corr_arb['contracts_deployed']/corr_arb['txs']).mean(),3)
plt.text(datetime.datetime(2021,10,1),0.06,f"Mean: {avg_percenttxscontracts_arb}")


# In[ ]:





# In[96]:


plt.figure()
cormat_arb = corr_arb.corr()
round(cormat_arb,2)
sns.heatmap(cormat_arb,annot=True)
plt.title('Arbitrum: Correlation Heatmap')


# In[97]:


cormat_arb


# In[179]:


zksync_gasprice


# In[181]:


zksync_txs


# In[121]:


#zkSync
deploycontracts_zksync = contracts_zksync.copy()
deploycontracts_zksync = deploycontracts_zksync.groupby(deploycontracts_zksync['block_time']).agg({'count'})
deploycontracts_zksync = deploycontracts_zksync.reset_index()
#deploycontracts_zksync['block_time'] = pd.to_datetime(deploycontractscount_poly['timestamp'], format="%Y-%m-%d")
deploycontracts_zksync['deployed_contracts_count'] = deploycontracts_zksync[('contract_address','count')]

avgdeploycontractscount_zksync = deploycontracts_zksync['deployed_contracts_count'].sum()/len(deploycontracts_zksync['deployed_contracts_count'])
stddeploycontractscount_zksync = round(deploycontracts_zksync['deployed_contracts_count'].std())



# In[122]:


deploycontracts_zksync


# In[123]:


deploycontracts_zksync['block_time'].iloc[0]


# In[131]:


contracts_zksync


# # Protocol Activity

# In[311]:


def plot_protocoldistribution(contracts_df, chain_name):
    print(contracts_df['namespace'].value_counts())
    
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=contracts_df['block_time'],
                                 columns=contracts_df['namespace'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab20',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=2,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel("Protocols (by proportion)")
    plt.title(f"{chain_name}: Distribution of Protocols")


# In[327]:


def plot_topn_protocoldistribution(contracts_df, chain_name, topn):
    topprotocols = list(contracts_df['namespace'].value_counts().keys()[:topn])
    temp_contracts_df = contracts_df[contracts_df['namespace'].isin(topprotocols)]
    print(temp_contracts_df['namespace'].value_counts())
    print(f"Top protocols account for {len(temp_contracts_df)/len(contracts_df)} of activity")
    
    plt.figure(figsize=(50,30))
    cross_tab_prop = pd.crosstab(index=temp_contracts_df['block_time'],
                                 columns=temp_contracts_df['namespace'],
                                 normalize="index")

    cross_tab_prop.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab20',
                        #legend=None,
                        figsize=(15, 10))

    plt.legend(loc="center left", ncol=2,bbox_to_anchor=(1, 0.5))
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel(f"Top {topn} Protocols (by proportion)")
    plt.title(f"{chain_name}: Distribution of Top {topn} Protocols")


# In[329]:


def plot_protocolmarketshare(contracts_df, chain_name, topn):
    norm = [float(i)/sum(contracts_df['namespace'].value_counts()[:topn]) for i in contracts_df['namespace'].value_counts()[:topn]]
    norm = list(map(lambda x: round(x*100,1),norm))
    norm_dict = {k: v for k,v in list(zip(contracts_df['namespace'].value_counts()[:topn].index,norm))}

    plt.figure()
    plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values())
    plt.title(f'{chain_name}: Protocols Transaction Share')
    plt.xticks(rotation=90)
    plt.xlabel('Protocol')
    plt.ylabel('Txs Share (%)')
    plt.bar_label(plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values()));
    


# In[87]:


def plot_protocolactivity(contracts_df, chain_name, topn,yscale="linear"):
    plt.figure(figsize=(30, 20))
    for p in contracts_df['namespace'].value_counts()[:topn].index:
        temp = contracts_df[contracts_df['namespace']==p]
        temp_count = temp.groupby('block_time').agg({'count'})
        plt.plot(temp_count.index,temp_count[('tx_hash','count')],label=p)
        plt.yscale(yscale)
    plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5),fontsize=20)
    plt.title(f'{chain_name}: Top {topn} Protocols Activity Over Time',fontsize=30)
    plt.xticks(rotation=90)
    plt.xlabel('Days',fontsize=20)
    plt.ylabel('Txs',fontsize=20);


# In[320]:


contracts_op[contracts_op['namespace'].isin(list(contracts_op['namespace'].value_counts().keys()[:20]))]


# In[321]:


contracts_op


# In[303]:


plot_protocoldistribution(contracts_op,'Optimism')


# In[328]:


plot_topn_protocoldistribution(contracts_op,'Optimism',20)


# In[330]:


plot_protocolmarketshare(contracts_op,'Optimism',20)


# In[332]:


plot_protocolactivity(contracts_op,'Optimism',20)


# In[312]:


plot_protocoldistribution(contracts_zksync,"zkSync Era")


# In[98]:


#Polygon
#!!!need to drop duplicates, order by time, etc.
#path = "/Users/cyrusleung/Desktop/Scroll/Chains Data/polygon txs/protocols"
#all_files = glob.glob(os.path.join(path, "*.csv"))
poly_protocols = pd.read_csv('/Users/cyrusleung/Desktop/Scroll/Chains Data/polygon txs/protocols/protocols_5-30_9-9.csv')


# In[99]:


poly_protocols['block_time'] = poly_protocols['block_time'].map(lambda x: x.split()[0])
poly_protocols['block_time'] = pd.to_datetime(poly_protocols['block_time'],format='%Y-%m-%d')


# In[100]:


poly_protocols


# In[101]:


poly_protocols.sort_values(by='block_time',ascending=True,inplace=True)
poly_protocols = poly_protocols.drop_duplicates(subset=['hash'], keep='first')


# In[102]:


poly_protocols


# In[103]:


poly_protocols['namespace'].value_counts()


# In[104]:


cross_tab_prop = pd.crosstab(index=poly_protocols['block_time'],
                             columns=poly_protocols['namespace'],
                             normalize="index")


# In[105]:


cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(15, 10))

plt.legend(loc="center left", ncol=2,bbox_to_anchor=(1, 0.5))
plt.xlabel("Date")
plt.ylabel("Protocols (by proportion)")
plt.title('Polygon: Distribution of Protocols')


# In[106]:


plt.figure(figsize=(30, 20))
for p in poly_protocols['namespace'].value_counts().index:
    temp = poly_protocols[poly_protocols['namespace']==p]
    temp_count = temp.groupby('block_time').agg({'count'})
    plt.plot(temp_count.index,temp_count[('hash','count')],label=p)
plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5),fontsize=20)
plt.title('Polygon: Protocols Activity Over Time',fontsize=30)
plt.xticks(rotation=90)
plt.xlabel('Days',fontsize=20)
plt.ylabel('Txs',fontsize=20);


# In[107]:


#Arbitrum
'''path = "/Users/cyrusleung/Desktop/Scroll/Chains Data/arbitrum txs/protocols"
all_files = glob.glob(os.path.join(path, "*.csv"))
arb_protocols = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
arb_protocols['block_time'] = arb_protocols['block_time'].map(lambda x: x.split()[0])
arb_protocols['block_time'] = pd.to_datetime(arb_protocols['block_time'],format='%Y-%m-%d')

arb_protocols.sort_values(by='block_time',ascending=True,inplace=True)
arb_protocols = arb_protocols.drop_duplicates(subset=['hash'], keep='first')

arb_protocols.to_csv('/Users/cyrusleung/Desktop/Scroll/Chains Data/arbitrum txs/protocols/all_protocols.csv',index=False)


# In[108]:


arb_protocols = pd.read_csv('/Users/cyrusleung/Desktop/Scroll/Chains Data/arbitrum txs/protocols/all_protocols.csv')


# In[109]:


arb_protocols


# In[6]:


arb_protocols['namespace'].value_counts()[:20].index


# In[111]:


cross_tab_prop = pd.crosstab(index=arb_protocols['block_time'],
                             columns=arb_protocols['namespace'],
                             normalize="index")


# In[112]:


cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab20', 
                    figsize=(30, 20))

plt.legend(loc="center left", ncol=2,bbox_to_anchor=(1, 0.5))
plt.xlabel("Date",fontsize=30)
plt.ylabel("Protocols (by proportion)",fontsize=30)
plt.title('Arbitrum: Distribution of Protocols',fontsize=30)


# In[113]:


arb_protocols_top20 = arb_protocols[arb_protocols['namespace'].isin(arb_protocols['namespace'].value_counts()[:20].index)]


# In[114]:


cross_tab_prop = pd.crosstab(index=arb_protocols_top20['block_time'],
                             columns=arb_protocols_top20['namespace'],
                             normalize="index")


# In[115]:


cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab20', 
                    figsize=(30, 20))

plt.legend(loc="center left", ncol=1,bbox_to_anchor=(1, 0.5))
plt.xlabel("Date",fontsize=30)
plt.ylabel("Protocols (by proportion)",fontsize=30)
plt.title('Arbitrum: Distribution of Top 20 Protocols',fontsize=30)


# In[116]:


sum(arb_protocols['namespace'].value_counts()[:20])/sum(arb_protocols['namespace'].value_counts())


# In[117]:


arb_protocols['namespace'].value_counts()[:20]


# In[118]:


norm = [float(i)/sum(arb_protocols['namespace'].value_counts()[:20]) for i in arb_protocols['namespace'].value_counts()[:20]]


# In[119]:


norm = list(map(lambda x: round(x*100,1),norm))


# In[120]:


norm


# In[121]:


norm_dict = {k: v for k,v in list(zip(arb_protocols['namespace'].value_counts()[:20].index,norm))}


# In[122]:


norm_dict.values()


# In[135]:


plt.figure()
plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values())
plt.title('Arbitrum: Protocols Transaction Share')
plt.xticks(rotation=90)
plt.xlabel('Protocol')
plt.ylabel('Txs Share (%)')
plt.bar_label(plt.bar(norm_dict.keys(),norm_dict.values(),label=norm_dict.values()));


# In[124]:


arb_protocols['namespace'].value_counts()[:20].index


# In[125]:


len(arb_protocols['namespace'].value_counts())


# In[126]:


sum(arb_protocols['namespace'].value_counts()[:10])/sum(arb_protocols['namespace'].value_counts())


# In[127]:


#Plot the transaction counts of each of the top 20 protocols over time
sushi = arb_protocols[arb_protocols['namespace']=='sushi']


# In[128]:


sushi_count = sushi.groupby('block_time').agg({'count'})


# In[129]:


plt.plot(sushi_count.index,sushi_count[('hash','count')],label='sushi')


# In[70]:


arb_protocols


# In[84]:


arb_protocols['timestamp'] = arb_protocols['timestamp'].map(clean_timestamp_transpose)


# In[16]:


arb_protocols['protocol_name'].value_counts()[:20]


# In[ ]:





# In[26]:


plt.figure()
for p in arb_protocols['namespace'].value_counts()[:20].index:
    temp = arb_protocols[arb_protocols['namespace']==p]
    temp_count = temp.groupby('block_time').agg({'count'})
    plt.plot(temp_count.index,temp_count[('hash','count')],label=p)
plt.legend(loc="center right", ncol=1, bbox_to_anchor=(1.5, 0.5))
plt.title('Arbitrum: Top 20 Protocols Activity Over Time')
plt.xticks(rotation=90)
plt.yscale("log")
'''plt.xlabel('Days',fontsize=20)
plt.ylabel('Txs',fontsize=20);'''


# In[ ]:


#What is the correlation between each protocol and 

