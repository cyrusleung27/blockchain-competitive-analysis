# blockchain-competitive-analysis

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
