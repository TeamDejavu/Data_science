#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import datetime


# In[16]:
df = pd.read_csv('./Dataset/PricingData.csv')


# In[17]:

df.info()


# In[18]:


df.isnull().sum()


# In[19]:


df.fillna(-1,inplace=True)


# In[20]:


df.drop_duplicates(inplace = True)


# In[21]:


f1 = df["Seat Fare Type 1"].values
for i in range(len(f1)):
    if(f1[i]==-1):
        continue
    temp = list(map(float,f1[i].split(",")))
    f1[i] = sum(temp)/len(temp)
    
f2 = df["Seat Fare Type 2"].values
for i in range(len(f2)):
    if(f2 [i]==-1):
        continue
    temp = list(map(float,f2[i].split(",")))
    f2[i] = sum(temp)/len(temp)
    


# In[22]:


sd = df["Service Date"].values
for i in range(len(sd)):
    sd[i] = datetime.datetime.strptime(sd[i],"%d-%m-%Y %H:%M")

rd = df["RecordedAt"].values
for i in range(len(rd)):
    rd[i] = datetime.datetime.strptime(rd[i],"%d-%m-%Y %H:%M")


# In[23]:


hour = []
for i in range(len(rd)):
    diff = sd[i] - rd[i]
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    hour.append(hours)
df['Hours'] = np.array(hour)


# In[24]:


df


# In[25]:


cost1 = dict()
cost2 = dict()
occ = dict()

for x in df["Bus"].values:
    if x not in occ.keys():
        occ[x] = 0
    else:
        occ[x] += 1

for i in range(len(df["Bus"].values)):
    try:
        if(df["Seat Fare Type 1"][i] != -1):
            try:
                if(df["Bus"][i] not in cost1.keys()):
                    cost1[df["Bus"][i]] = int(df["Seat Fare Type 1"][i])
                else:
                    cost1[df["Bus"][i]] += int(df["Seat Fare Type 1"][i])
            except:
                continue
    except:
        continue
        
for i in range(len(df["Bus"].values)):
    try:
        if(df["Seat Fare Type 2"][i] != -1):
            try:
                if(df["Bus"][i] not in cost2.keys()):
                    cost2[df["Bus"][i]] = int(df["Seat Fare Type 2"][i])
                else:
                    cost2[df["Bus"][i]] += int(df["Seat Fare Type 2"][i])
            except:
                continue
    except:
        continue


# In[26]:


cost2


# In[27]:


for key in cost1.keys():
    #print(key, cost1[key],occ[key])
    try:
        cost1[key] /= occ[key]
    except:
        continue
        
for key in cost2.keys():
    try:
        cost2[key] /= occ[key]
    except:
        continue


# In[28]:


from sklearn.cluster import KMeans


# In[29]:


encoding = dict()
num = 0
for x in df["Bus"].values:
    if x not in encoding.keys():
        encoding[x] = num
        num+=1
encoding


# In[30]:


def change_v(x):
    return encoding[x]


# In[31]:


df["Bus"] = df["Bus"].apply(change_v)


# In[32]:


df


# In[33]:


x = df.iloc[:,[0,1,2,5]].values
x


# In[34]:


kmeans = KMeans(n_clusters=117,random_state = 3).fit(x)


# In[35]:


c1,c2 = 0,0
s1,s2 = 0,0
for x in df['Seat Fare Type 1'].values:
    if(x == -1):
        continue
    s1+=x
    c1+=1
for x in df['Seat Fare Type 2'].values:
    if(x == -1):
        continue
    s2+=x
    c2+=1
s1 /= c1
s2 /= c2


# In[36]:


def func(bus):
    bus = bus.strip()
    try:
        ss1 = cost1[bus]
        ss2 = cost2[bus]
    except:
        ss1 = s1
        ss2 = s2
    res_li = []
    def max_occ(seq):
        from operator import itemgetter
        c = dict()
        for item in seq:
            c[item] = c.get(item, 0) + 1
        return max(c.items(), key=itemgetter(1))
    
    def get_key(value):
        for name, age in encoding.items():
            if age == value:
                return name
    
    def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]
        
    code = encoding[bus]
    
    ans = kmeans.predict([[ss1,ss2,code,0]])
    ar = ClusterIndicesNumpy(ans[0], kmeans.labels_)
    li = []
    for x in ar:
        try:
            li.append(df.loc[x,["Bus"]].values[0])
        except:
            continue
    m = max_occ(li)
    bus_res1 = get_key(m[0])
    bus_conf1 = m[1]/len(li)
    res_li.append(bus_res1)
    res_li.append(bus_conf1)
    
    return res_li
    
    


# In[37]:


url = "https://drive.google.com/file/d/1l9AK2ISBMz6Rqqre6lwU9RTQKTx0tHlu/view?usp=sharing"
path = path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df_out = pd.read_csv(path)


# In[42]:


df_out


# In[39]:


conf_dict = dict()
res_dict = dict()
for key in occ.keys():
    temp = func(key)
    res_dict[key] = temp[0]
    conf_dict[key] = temp[1]
res_dict


# In[40]:


for i in range(len(df_out.values)):
    df_out["Follows"][i] = res_dict[df_out.loc[i,"Bus"]]
    df_out["Confidence Score (0 to 1)"][i] = conf_dict[df_out.loc[i,"Bus"]]
    df_out["Is followed by"][i] = res_dict[res_dict[df_out.loc[i,"Bus"]]]
    df_out["Confidence Score (0 to 1).1"][i] = conf_dict[res_dict[df_out.loc[i,"Bus"]]]


