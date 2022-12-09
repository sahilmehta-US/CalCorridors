#!/usr/bin/env python
# coding: utf-8

# ### Import Statements

# In[1]:


import matplotlib as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint
from matplotlib import pyplot as plt
import fiona
import os
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
import time


# ### Data Reading and Parsing

# In[2]:


filepath1 = "ds2734/ds2734.gdb"
filepath2 = "fire21_2.gdb"
f = fiona.open(filepath2)
FireData = gpd.GeoDataFrame.from_features([feature for feature in f], crs='epsg:3310')
ConnectivityData = gpd.read_file(filepath1).to_crs('epsg:3310')
HabitatDataList = gpd.GeoDataFrame({'ecosystem':[]})
namelist = []
filepaths = os.path.basename("Monier/CWHR_Habitats")
SpeciesDataList = None

for filename in os.listdir(filepaths):
    f = os.path.join(filepaths, filename)
    if os.path.isfile(f) and os.path.basename(f)[-3:] == "shp":
        dset = gpd.read_file(f).to_crs('epsg:3310')
        dset['ecosystem'] = os.path.basename(f)[:-4]
        namelist.append(dset['ecosystem'][0])
        HabitatDataList = pd.concat([HabitatDataList, dset])
HabitatDataList.crs = 'epsg:3310'
HabitatDataList = HabitatDataList.reset_index()
ConnectivityDataFull = gpd.read_file('ConnectivityFull.geojson')


# In[70]:


ax = ConnectivityData.plot(figsize = (100,50), color = 'blue')
ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 5].plot(ax = ax, color = 'green')
FireData.plot(ax = ax, color = 'red')
HabitatDataList.boundary.plot(ax = ax)


# In[71]:


ax = ConnectivityData.plot(figsize = (100,50), color = 'green')
ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 5].plot(ax = ax, color = 'blue')
FireData.plot(ax = ax, color = 'red')
HabitatDataList.boundary.plot(ax = ax)


# In[ ]:


arr = np.zeros(58)
arr2 = np.zeros(58)
# for node in ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 5]['geometry']:
#     i = 0
#     for habitat in HabitatDataList['geometry']:
#         area = habitat.intersection(node).area
#         if area < node.area and area > 0:
#             arr[i] += 1
#             break
#         i += 1
for node in ConnectivityData.iloc:
    i = 0
    for habitat in HabitatDataList['geometry']:
        area = habitat.intersection(node['geometry']).area
        if area < node['geometry'].area and area > 0:
            if node['Connectivity_rank'] == 5:
                arr[i] += 1
            arr2[i] += 1
            break
        i += 1
frame1 = pd.DataFrame({'Ecosystem': HabitatDataList['ecosystem'], 'num_cat5_nodes':arr, 'num_nodes':arr2})
frame1.to_file('frame1.csv', driver = 'CSV')


# In[7]:


area = 6474000
pdatafr = gpd.GeoDataFrame({})
for i in range(len(ConnectivityData)):
    if ConnectivityData['geometry'][i].area < area:
        stuff = ConnectivityData.iloc[i]
        arr = []
        for col in stuff.index:
            arr.append(tuple([col, stuff[col]]))
        dataf = gpd.GeoDataFrame({arr[0][0]: arr[0][1]}, {0})
        for tup in arr[1:]:
            dataf = dataf.join(gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0}), lsuffix = tup[0])
        pdatafr = pd.concat([pdatafr, dataf])
pdatafr = gpd.GeoDataFrame(pdatafr)
pdatafr = pdatafr.reset_index()
pdatafr.crs = 'epsg:3310'
ConnectivityDataFull = pd.concat([ConnectivityData, pdatafr]).drop_duplicates(keep = False)
ConnectivityDataFull.to_file('ConnectivityFull.geojson', driver = 'GeoJSON')


# In[3]:


clusters = []
clusterframes = []
centroidDist = 2750
cdFullR5 = ConnectivityDataFull.loc[ConnectivityDataFull['Connectivity_rank'] == 5]
for i in range(len(cdFullR5)):
    tic = time.perf_counter()
    poly = cdFullR5.iloc[i]
    arr = []
    for col in poly.index:
        arr.append(tuple([col, poly[col]]))
    polyf = gpd.GeoDataFrame({arr[0][0]: arr[0][1]}, {0})
    for tup in arr[1:]:
        polyf = polyf.join(gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0}), lsuffix = tup[0])
    polyloc = -1
    if poly['geometry'] in clusters:
        polyloc = np.where(cluster == poly['geometry'])[0][0]
    else:
        clusterframes.append(polyf)
        clusters.append([poly])
        polyloc = len(clusters)-1
    for j in range(i+1, len(cdFullR5)):
        poly2 = cdFullR5.iloc[j]
        arr = []
        for col in poly2.index:
            arr.append(tuple([col, poly2[col]]))
        polyf2 = gpd.GeoDataFrame({arr[0][0]: arr[0][1]}, {0})
        for tup in arr[1:]:
            polyf2 = polyf2.join(gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0}), lsuffix = tup[0])
        if poly2['geometry'].centroid.distance(poly['geometry'].centroid) < centroidDist:
            clusters[polyloc].append(poly2)
            clusterframes[polyloc] = pd.concat([clusterframes[polyloc], polyf2])
    toc = time.perf_counter()
    print(str(100*(i+1)/len(cdFullR5)) + f"% done in time {toc - tic:0.4f} seconds")
for i in range(len(clusterframes)):
    clusterframes[i] = gpd.GeoDataFrame(clusterframes[i])
    clusterframes[i].crs = 'epsg:3310'


# In[4]:


for i in range(len(clusterframes)):
    clusterframes[i] = gpd.GeoDataFrame(clusterframes[i])
    clusterframes[i].crs = 'epsg:3310'
    clusterframes[i] = clusterframes[i].reset_index()


# In[5]:


arr = []
i = 0
for dataframe in clusterframes:
    area = 0
    for geom in dataframe['geometry']:
        for region in FireData['geometry']:
            area += geom.intersection(region).area
    arr.append(100*area/(6*sum(dataframe['geometry'].area)))
    i+=1
clusterframes2 = np.concatenate((np.array(clusterframes)[:, None], np.array(arr)[:, None]), axis = 1)


# In[25]:


for i in clusterframes2:
    print(i[1])


# In[65]:


Linkages = pd.DataFrame({'eco1': [], 'eco2': [], 'num': []})
Chord_format = pd.DataFrame({'source': [], 'target':[], 'value':[]})
for i in range(len(HabitatDataList['ecosystem'])):
    for j in range(i+1, len(HabitatDataList['ecosystem'])):
        ecosystem = list(HabitatDataList['ecosystem'])[i]
        ecosystem2 = list(HabitatDataList['ecosystem'])[j]
        count = 0
        for database in clusterframes:
            polygon = database['geometry'][0]
            for x in range(1, len(database)):
                polygon = polygon.union(database['geometry'][x])
            if HabitatDataList.iloc[i]['geometry'].intersects(polygon) and HabitatDataList.iloc[j]['geometry'].intersects(polygon):
                count += 1
        df = pd.DataFrame({'eco1': [ecosystem], 'eco2': [ecosystem2], 'num':[count]})
        df2 = pd.DataFrame({'source': [int(i)], 'target':[int(j)], 'value': [int(count)]})
        Linkages = pd.concat([Linkages, df])
        if count > 0:
            Chord_format = pd.concat([Chord_format, df2])
    Linkages.reset_index().drop(columns = ['index'])
#Linkages.to_file('linkages.geojson', driver = 'GeoJSON')
# chord_from_df = Chord(Linkages, source="eco1", target="eco2", value="num")
# show(chord_from_df)
Chord_format = Chord_format.reset_index().drop('index', axis = 1)
Chord_format['source'] = np.array(Chord_format['source'], dtype = int)
Chord_format['target'] = np.array(Chord_format['target'], dtype = int)
Chord_format['value'] = np.array(Chord_format['value'][i], dtype = int)


hv.extension('bokeh')
hv.output(size = 300)
nodes = hv.Dataset(pd.DataFrame(HabitatDataList['ecosystem']), 'index')
chord = hv.Chord((Chord_format,nodes))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap = 'Category20', edge_color=dim('source').str(), 
               labels='ecosystem', node_color=dim('index').str()))


# In[66]:


hv.extension('bokeh')
hv.output(size = 300)
nodes = hv.Dataset(pd.DataFrame(HabitatDataList['ecosystem']), 'index')
chord = hv.Chord((Chord_format,nodes))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap = 'Category20', edge_color=dim('source').str(), 
               labels='ecosystem', node_color=dim('index').str()))


# In[67]:


import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
hv.extension('bokeh')

linksExample = pd.DataFrame(data['links'])
nodesExample = hv.Dataset(pd.DataFrame(data['nodes']), 'index')

chord = hv.Chord((linksExample, nodesExample)).select(value=(5, None))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))


# In[63]:


Chord_format['source'] = np.array(Chord_format['source'], dtype = int)
Chord_format['target'] = np.array(Chord_format['target'], dtype = int)
Chord_format['value'] = np.array(Chord_format['value'][i], dtype = int)


# In[62]:


Chord_format = Chord_format.reset_index().drop('index', axis = 1)


# In[147]:


#For resetting indexes and dropping added column:
#    Chord_format = Chord_format.reset_index().drop('index', axis = 1)


#For series -> geodataframe

# arr = []
# for col in stuff.index:
#     arr.append(tuple([col, stuff[col]]))
# dataf = gpd.GeoDataFrame({arr[0][0]: arr[0][1]}, {0})
# for tup in arr[1:]:
#     gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0})
#     dataf = dataf.join(gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0}), lsuffix = tup[0])


# In[203]:


ax = ConnectivityData.plot(figsize = (100,50), color = 'blue')
ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 5].plot(ax = ax, color = 'green')
ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 4].plot(ax = ax, color = 'red')


# In[ ]:




