#!/usr/bin/env python
# coding: utf-8

# ### Converters

# In[ ]:


#For resetting indexes:
#    geodataframe = geodataframe.reset_index()

#For dropping column
# geodataframe = geodataframe.drop("index", axis = 1)

#For series -> geodataframe

# arr = []
# for col in stuff.index:
#     arr.append(tuple([col, stuff[col]]))
# dataf = gpd.GeoDataFrame({arr[0][0]: arr[0][1]}, {0})
# for tup in arr[1:]:
#     gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0})
#     dataf = dataf.join(gpd.GeoDataFrame({tup[0]: [tup[1]]}, {0}), lsuffix = tup[0])


# ### Import Statements

# In[245]:


import matplotlib as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import shapely
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint
from matplotlib import pyplot as plt
import fiona
import os
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
import time
import networkx as nx
import random
import math


# ### Data Reading and Parsing

# In[15]:


G = nx.Graph()
G.add_node('EOR', weight = 2)
G.add_node('ABC', weight = 1)
G.add_node('DEF', weight = 0)
G.add_nodes_from([('XYZ', {'weight': 4}),('123', {'weight': 3})])
G.add_edge('EOR','ABC', weight = 4)
G.add_edges_from([('EOR','DEF', {'weight' : 3}),('DEF','XYZ', {'weight' : 4}), ('XYZ','ABC', {'weight' : 3}), ('123','DEF',{'weight' : 2})])
labels = nx.get_edge_attributes(G,'weight')
weights = nx.get_node_attributes(G, 'weight')
pos = weights.copy()
for i in weights:
    pos[i] = ((random.randint(3,9), random.randint(3,9)))
nx.draw(G, pos, with_labels = True)
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)


# In[17]:


pos


# In[64]:


filepath1 = "ds2734/ds2734.gdb"
filepath2 = "fire21_2.gdb"
filepath3 = "Cal Red Frog/ds246.gdb"
f = fiona.open(filepath2)
FireData = gpd.GeoDataFrame.from_features([feature for feature in f], crs='epsg:3310')
ConnectivityData = gpd.read_file(filepath1).to_crs('epsg:3310')
Red_Frog_Habitat = gpd.read_file(filepath3).to_crs('epsg:3310')
HabitatDataList = gpd.GeoDataFrame({'ecosystem':[]})
namelist = []
filepaths = os.path.basename("Monier/CWHR_Habitats")
SpeciesDataList = None
for filename in os.listdir(filepaths):
    f = os.path.join(filepaths, filename[2:])
    if os.path.isfile(f) and os.path.basename(f)[-3:] == "shp":
        dset = gpd.read_file(f).to_crs('epsg:3310')
        dset['ecosystem'] = os.path.basename(f)[:-4]
        namelist.append(dset['ecosystem'][0])
        HabitatDataList = pd.concat([HabitatDataList, dset])

# count = 0
# filepaths2 = "CWHR_Species_Ranges/Amphibians"
# for file in os.listdir(filepaths2):
#     if file[-3:] == "shp":
#         dset = gpd.read_file(filepaths2 + "/" + file).to_crs('epsg:3310')
#         if count == 0:
#             count+=1
#             AmphibianDataList = dset
#         else:
#             AmphibianDataList = pd.concat([AmphibianDataList, dset])
            
# filepaths3 = "CWHR_Species_Ranges/Birds"
# count = 0
# for file in os.listdir(filepaths3):
#     if file[-3:] == "shp":
#         dset = gpd.read_file(filepaths3 + "/" + file).to_crs('epsg:3310')
#         if count == 0:
#             count+=1
#             BirdDataList = dset
#         else:
#             BirdDataList = pd.concat([BirdDataList, dset])
            
# filepaths4 = "CWHR_Species_Ranges/Mammals"
# count = 0
# for file in os.listdir(filepaths4):
#     if file[-3:] == "shp":
#         dset = gpd.read_file(filepaths4 + "/" + file).to_crs('epsg:3310')
#         if count == 0:
#             count+=1
#             MammalDataList = dset
#         else:
#             MammalDataList = pd.concat([MammalDataList, dset])

# filepaths5 = "CWHR_Species_Ranges/Reptiles"
# count = 0
# for file in os.listdir(filepaths5):
#     if file[-3:] == "shp":
#         dset = gpd.read_file(filepaths5 + "/" + file).to_crs('epsg:3310')
#         if count == 0:
#             count+=1
#             ReptileDataList = dset
#         else:
#             ReptileDataList = pd.concat([ReptileDataList, dset])

# AmphibianDataList = AmphibianDataList.reset_index().drop('index', axis = 1).drop('Season', axis = 1).drop('SHAPE_NAME', axis = 1).drop('SEASON', axis = 1).drop('OBJECTID', axis = 1).drop('Shape_Leng', axis = 1).drop('Shape_Area', axis = 1).drop('Sname', axis = 1).drop('Cname', axis = 1)
# MammalDataList = MammalDataList.reset_index().drop('index', axis = 1).drop('Season', axis = 1).drop('SEASON', axis = 1).drop("Shape_Leng", axis = 1).drop("Shape_Area", axis = 1)
# ReptileDataList = ReptileDataList.reset_index().drop('index', axis = 1).drop('Season', axis = 1).drop('SEASON', axis = 1).drop('Sname', axis = 1).drop('SHAPE_NAME', axis = 1).drop('Cname', axis = 1).drop('ACRES', axis = 1).drop('Notes', axis = 1)
# BirdDataList = BirdDataList.reset_index().drop('index', axis = 1).drop('Season', axis = 1).drop('SHAPE_NAME', axis = 1)
HabitatDataList.crs = 'epsg:3310'
HabitatDataList = HabitatDataList.reset_index()
ConnectivityDataFull = gpd.read_file('ConnectivityFull.geojson')
problems = []
for i in range(len(FireData)):
    if not isinstance(FireData['YEAR_'][i], str):
        problems.append(i)
FireData = FireData.drop(problems).reset_index().drop('index', axis = 1)


# ### Code Analysis

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


# In[171]:


speciesList = []
HabitatDataList['SpeciesNumbers'] = np.zeros(58)
i = 0
for habitat in HabitatDataList['geometry']:
    speciesList.append([])
    for j in range(len(BirdDataList['geometry'])):
        if habitat.intersection(BirdDataList['geometry'][j]).area > 0:
            speciesList[i].append(BirdDataList['CName'][j])
            HabitatDataList['SpeciesNumbers'][i] += 1
    i+=1
HabitatDataList['SpeciesList'] = speciesList


# In[172]:


speciesList = []
HabitatDataList['SpeciesNumbers'] = np.zeros(58)
i = 0
for habitat in HabitatDataList['geometry']:
    speciesList.append([])
#     for j in range(len(BirdDataList['geometry'])):
#         if habitat.intersection(BirdDataList['geometry'][j]).area > 0:
#             speciesList[i].append(BirdDataList['CName'][j])
#             HabitatDataList['SpeciesNumbers'][i] += 1
    for j in range(len(MammalDataList['geometry'])):
        if habitat.intersection(MammalDataList['geometry'][j]).area > 0:
            speciesList[i].append(MammalDataList['CName'][j])
            HabitatDataList['SpeciesNumbers'][i] += 1
#     for j in range(len(AmphibianDataList['geometry'])):
#         if habitat.intersection(AmphibianDataList['geometry'][j]).area > 0:
#             speciesList[i].append(AmphibianDataList['CName'][j])
#             HabitatDataList['SpeciesNumbers'][i] += 1
    for j in range(len(ReptileDataList['geometry'])):
        if habitat.intersection(ReptileDataList['geometry'][j]).area > 0:
            speciesList[i].append(ReptileDataList['CName'][j])
            HabitatDataList['SpeciesNumbers'][i] += 1
    i+=1
    print(i)
HabitatDataList['SpeciesList'] = speciesList


# In[173]:


HabitatDataList


# In[47]:


nodes = []
for i in range(len(HabitatDataList)):
    nodes.append((HabitatDataList['ecosystem'][i], {'weight': HabitatDataList['SpeciesNumbers'][i]}))


# In[9]:


clusterframes = gpd.read_file('clusterframes.geojson')
Linkages = pd.read_csv('linkages.csv')


# In[88]:


Linkages = pd.read_csv('Linkages_and_Fires.csv').drop('Unnamed: 0', axis = 1)
tot60s = 0
tot70s = 0
tot80s = 0
tot90s = 0
tot00s = 0
tot10s = 0
totar = 0
for i in range(len(HabitatDataList['ecosystem'])):
    ecosystem = HabitatDataList['ecosystem'][i]
    for j in range(i+1, len(HabitatDataList['ecosystem'])):
        ecosystem2 = HabitatDataList['ecosystem'][j]
        if ecosystem2 in list(Linkages.loc[Linkages['eco1'] == ecosystem]['eco2']):
            continue
        totarea = 0
        burnarea60s = 0
        burnarea70s = 0
        burnarea80s = 0
        burnarea90s = 0
        burnarea00s = 0
        burnarea10s = 0
        for k in range(len(clusterframes)):
            geo = clusterframes['geometry'][k]
            if HabitatDataList['geometry'][i].intersects(geo) and HabitatDataList['geometry'][j].intersects(geo):
                totarea += geo.area
                for l in range(len(FireData['geometry'])):
                    FireArea = FireData['geometry'][l]
                    if geo.intersects(FireArea):
                        if FireData['YEAR_'][l] in string60s:
                            burnarea60s += geo.intersection(FireData['geometry'][l]).area
                        if FireData['YEAR_'][l] in string70s:
                            burnarea70s += geo.intersection(FireData['geometry'][l]).area
                        if FireData['YEAR_'][l] in string80s:
                            burnarea80s += geo.intersection(FireData['geometry'][l]).area
                        if FireData['YEAR_'][l] in string90s:
                            burnarea90s += geo.intersection(FireData['geometry'][l]).area
                        if FireData['YEAR_'][l] in string00s:
                            burnarea00s += geo.intersection(FireData['geometry'][l]).area
                        if FireData['YEAR_'][l] in string10s:
                            burnarea10s += geo.intersection(FireData['geometry'][l]).area
        if totarea == 0:
            num = None
        else:
            num60s = 100 - 100*burnarea60s/totarea
            num70s = 100 - 100*burnarea70s/totarea
            num80s = 100 - 100*burnarea80s/totarea
            num90s = 100 - 100*burnarea90s/totarea
            num00s = 100 - 100*burnarea00s/totarea
            num10s = 100 - 100*burnarea10s/totarea
        if i == 1:
            totar += totarea
        tot60s += burnarea60s
        tot70s += burnarea70s
        tot80s += burnarea80s
        tot90s += burnarea90s
        tot00s += burnarea00s
        tot10s += burnarea10s
        df = pd.DataFrame({'eco1': [ecosystem], 'eco2': [ecosystem2], 'percent60s': [num60s], 'percent70s': [num70s], 'percent80s': [num80s], 'percent90s': [num90s], 'percent00s': [num00s], 'percent10s': [num10s]})
        Linkages = pd.concat([Linkages, df])
    print(100*i/58)


# In[89]:


Linkages.to_csv('Linkages_and_Fires.csv')


# In[113]:


HabitatDataList


# In[164]:


health60s = 0
health70s = 0
health80s = 0
health90s = 0
health00s = 0
health10s = 0
maxhealth = 0
for i in range(len(Linkages)):
    num = int(HabitatDataList.loc[HabitatDataList['ecosystem'] == Linkages['eco1'][i]]['SpeciesNumbers']) + int(HabitatDataList.loc[HabitatDataList['ecosystem'] == Linkages['eco2'][i]]['SpeciesNumbers'])
    health60s += num*Linkages['percent60s'][i]
    health70s += num*Linkages['percent70s'][i]
    health80s += num*Linkages['percent80s'][i]
    health90s += num*Linkages['percent90s'][i]
    health00s += num*Linkages['percent00s'][i]
    health10s += num*Linkages['percent10s'][i]
    maxhealth += num*100


# In[201]:


health['health']


# In[202]:


plt.xlabel('year')
plt.ylabel('health')
ax = plt.plot('year', 'health', data=health)


# In[162]:


increasing = Linkages.copy()
decreasing = Linkages.copy()
for i in range(len(Linkages)):
    m,b = np.polyfit([1960,1970,1980,1990,2000,2010], list(Linkages.iloc[i])[2:], deg=1)
    if not (m > 0):
        increasing = increasing.drop(i)
    else:
        decreasing = decreasing.drop(i)


# In[174]:


increasing = increasing.reset_index().drop('index', axis = 1)
decreasing = decreasing.reset_index().drop('index', axis = 1)


# In[194]:


top10 = pd.DataFrame({'eco1': [], 'eco2': []})
locations = []
location = []
diffs = []
while(len(location) < 10):
    dec2 = decreasing.copy()
    for obj in locations:
        dec2 = dec2.drop(obj).reset_index().drop('index', axis = 1)
    diff = 0
    locator = 0
    for i in range(len(dec2)):
        num = int(HabitatDataList.loc[HabitatDataList['ecosystem'] == dec2['eco1'][i]]['SpeciesNumbers']) + int(HabitatDataList.loc[HabitatDataList['ecosystem'] == dec2['eco2'][i]]['SpeciesNumbers'])
        m,b = np.polyfit([1960,1970,1980,1990,2000,2010], list(dec2.iloc[i])[2:], deg=1)
        difference = (num*(b+2010*m)-num*100)
        if difference < diff:
            locator = i
            diff = difference
    if diff in diffs:
        locations.append(locator)
        continue
    diffs.append(diff)
    location.append(locator)
    locations.append(locator)
    top10 = pd.concat([top10, pd.DataFrame({'eco1': [dec2['eco1'][locator]], 'eco2': [dec2['eco2'][locator]]})])


# In[243]:


top10 = top10.reset_index().drop('index', axis = 1)


# In[237]:


list(HabitatDataList.loc[HabitatDataList['ecosystem'] == 'ADS']['geometry'])[0]


# In[247]:


geometries = []
splitGeoms = []
for i in range(len(top10)):
    partgeoms = []
    eco1 = list(HabitatDataList.loc[HabitatDataList['ecosystem'] == top10['eco1'][i]]['geometry'])[0]
    eco2 = list(HabitatDataList.loc[HabitatDataList['ecosystem'] == top10['eco2'][i]]['geometry'])[0]
    for corridor in clusterframes['geometry']:
        if corridor.intersects(eco1) and corridor.intersects(eco2):
            partgeoms.append(corridor)
    splitGeoms.append(corridor)
    geometries.append(shapely.union_all(partgeoms))


# In[298]:


centroidDist = 2750
for i in range(len(geometries)):
    statement = True
    geom = geometries[i]
    while(statement):
        statement = False
        for poly in ConnectivityData.loc[ConnectivityData['Connectivity_rank'] == 5]['geometry']:
            if poly.intersects(geom):
                areabefore = geom.area
                geom = shapely.union(geom, poly)
                statement=True
                if areabefore == geom.area:
                    statement = False
    geometries[i] = geom
    print('donezo')


# In[294]:


len(geometries)


# In[300]:


top10geoms['geometry'] = geometries


# In[309]:


FireData.loc[FireData['YEAR_'] == '201'].plot()


# In[313]:


plt.xlabel('year')
plt.ylabel('Burn Area (Standardized)')
plt.plot([1970,1980,1990,2000,2010,2020], burns2['burn_areas'])


# In[324]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot([1970,1980,1990,2000,2010,2020], health['health'], 'b-')
ax2.plot([1970,1980,1990,2000,2010,2020], burns2['burn_areas'], 'r-')
ax1.set_xlabel('Year')
ax1.set_ylabel('Health (Standardized)', color='b')
ax2.set_ylabel('Burn Area (Standardized)', color='r')
print('correlation value is: ' + str(r_value))


# In[323]:


plt.xlabel('burn area')
plt.ylabel('health')
plt.plot(burns2['burn_areas'],health['health'], 'o')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(burns2['burn_areas'],health['health'])
plt.plot(np.arange(.6,1.8,.01), intercept+slope*np.arange(.6,1.8,.01))
print('correlation value is: ' + str(r_value))


# In[226]:


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(burns2['burn_areas'],health['health'])


# In[227]:


r_value


# In[45]:


#Linkages = pd.DataFrame({'eco1': [], 'eco2': [], 'percent' : []})
for i in range(len(HabitatDataList['ecosystem'])):
    ecosystem = HabitatDataList['ecosystem'][i]
    if ecosystem in list(Linkages['eco1']):
        continue
    for j in range(i+1, len(HabitatDataList['ecosystem'])):
        ecosystem2 = HabitatDataList['ecosystem'][j]
        totarea = 0
        burnarea = 0
        for k in range(len(clusterframes)):
            geo = clusterframes['geometry'][k]
            if HabitatDataList['geometry'][i].intersects(geo) and HabitatDataList['geometry'][j].intersects(geo):
                totarea += geo.area
                for region in FireData['geometry']:
                    burnarea += geo.intersection(region).area
        if totarea == 0:
            num = None
        else:
            num = 100 - 100*burnarea/totarea
        df = pd.DataFrame({'eco1': [ecosystem], 'eco2': [ecosystem2], 'percent': [num]})
        Linkages = pd.concat([Linkages, df])


# In[152]:


plt.figure(figsize = (100,100))
i = 0
for eco in HabitatDataList['ecosystem']:
    color = '#' + hex(random.randint(16,255))[-2:] + hex(random.randint(16,255))[-2:] + hex(random.randint(16,255))[-2:]
    if (i == 0):
        i = 1
        ax = HabitatDataList.loc[HabitatDataList['ecosystem'] == eco].plot(color = 'white', edgecolor = color, figsize = (100,50))
    else: 
        HabitatDataList.loc[HabitatDataList['ecosystem'] == eco].plot(color = 'white', edgecolor = color, ax = ax)
plt.legend(list(HabitatDataList['ecosystem']), prop={'size': 200})


# In[162]:


Linkages.to_csv('linkages.csv')


# In[168]:


Links = Linkages.dropna().reset_index().drop('index', axis = 1)
edges = []
for i in range(len(Links)):
    edges.append((Links['eco1'][i], Links['eco2'][i], {'weight': Links['percent'][i]/100}))


# In[169]:


Links


# In[89]:


G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
edgelabels = nx.get_edge_attributes(G,'weight')
labels = {n: str(n) + ';' + str(int(G.nodes[n]['weight'])) for n in G.nodes}
pos = nx.get_node_attributes(G, 'weight').copy()
j = 0
for i in pos:
    pos[i] = ((50+50*math.cos(j*math.pi*2/len(pos)), 50+50*math.sin(j*math.pi*2/len(pos))))
    j += 1
plt.margins(x=1)
plt.figure(figsize = (100,100))
nx.draw(G, pos, with_labels = True, labels = labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels = edgelabels)


# In[72]:


ThreatenedReptileList = []
for i in range(len(ReptileDataList)):
    if ReptileDataList['SName'][i] in list(ThreatenedEndangered['Scientific Name']):
        ThreatenedReptileList.append(ReptileDataList['CName'][i])


# In[75]:


ThreatenedReptileList


# In[77]:


list(MammalDataList['CName'])


# In[68]:


ThreatenedEndangered = gpd.read_file('ECOS USFWS Threatened  Endangered Species Active Critical Habitat Report.csv')


# In[69]:


ThreatenedEndangered


# In[65]:


pd.set_option('display.max_rows', None)


# In[ ]:


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


# In[ ]:


clusterframes = gpd.read_file('clusterframes.geojson')


# In[ ]:


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


# In[27]:


cat5 = gpd.read_file('Cat5Nodes_per_Ecosystem.csv')


# In[ ]:


clusterframes = gpd.read_file()


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




