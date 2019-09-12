import numpy as np
import pandas as pd
import json 
from datetime import datetime, timedelta
from lib.data_preprocessing_lib import MapIndustrySector
from sklearn.externals import joblib

import json 

def validate_data(data):
    if type(data)==dict:
        return [data]
    elif type(data)==list:
        return data
    else:
        return None

def date_add(date, diff):
    if len(date)==10:
        dt_format = '%Y-%m-%d'
    elif len(date)==19:
        dt_format = '%Y-%m-%d %H:%M:%S'
    dt = datetime.strptime(date, dt_format) + timedelta(days=diff)
    return datetime.strftime(dt, dt_format)
    
def stringify_dict(data):
    data = validate_data(data)
    new_data = []
    for datum in data:
        for k in datum.keys():
            datum[k] = str(datum.get(k))
        new_data.append(datum)
    return new_data

def write_data_to_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
        
def read_data_from_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
        f.close()
    return data
        
def from_pandas_to_json(dataframe):
    data = []
    for idx in dataframe.index:
        datum = dict(dataframe.loc[idx])
        for k in datum.keys():
            datum[k] = str(datum.get(k))
        data.append(datum) 
    return data

def classify_fuel_consumption(data):
    # pre-cleanse the data
    industry_mapper = MapIndustrySector()
    data = validate_data(data)
    df = industry_mapper.transform(pd.DataFrame(data))

    #  geo-cluster and  equipment segment
    Xlonlat = df[['gps_longitude', 'gps_latitude']].astype(float).values
    geo_cluster_model = joblib.load("model/geo_location_clustering.pkl")
    geo_cluster = cluster_by_geolocation(clf=geo_cluster_model, Xlonlat=Xlonlat)
    df["geo_cluster"] = geo_cluster
    df["segment"] = df.apply(equipments_segmentation, axis=1)

    # make fuel scoring
    for segment in df['segment'].drop_duplicates().tolist():
        model_file = "model/{}_fuel_efficiency_index_model-latest.pkl".format(segment)
        pipeline = joblib.load(model_file)
        idx = df[df['segment']==segment].index
        result = pipeline.transform(df.loc[idx, :])
        df.loc[idx, "fuel_consumption_index"] = result[:,1]
        df.loc[idx, "traveling_anomaly"] = result[:,2]
    return from_pandas_to_json(df)

def join_dataframe(primary, secondaries, method, join_keys):
    merge = primary.copy()
    for dataframe in secondaries:
        merge = pd.merge(merge, dataframe, how=method, on=join_keys)
    return merge

def cluster_by_geolocation(clf, Xlonlat):
    Xlonlat = Xlonlat.astype(float)
    label = clf.predict(Xlonlat)
    
    # Separate Nusa Tenggara from East Java and Sulawesi
    nusa_idx = [i for i,pos in enumerate(Xlonlat) if
                (pos[0]>=114 and pos[0]<133 and # longitude boundaries
                 pos[1]>=-11 and pos[1]<-7.6)] # latitude boundaries
    label[nusa_idx] = clf.n_clusters
    
    # Ensure Lampung is separated from Java
    sumatera_label = get_cluster_id(clf, lon=105, lat=-2.5)
    lampung_idx = [i for i,pos in enumerate(Xlonlat) if 
                   (pos[0]>=102 and pos[0]<106 and # longitude boundaries
                    pos[1]>=-6 and pos[1]<-4)] # latitude boundaries
    label[lampung_idx] = sumatera_label
    
    # Ensure North Sulawesi is separated from East Kalimantan
    sulawesi_label = get_cluster_id(clf, lon=122, lat=-4)
    nort_sulawesi_idx = [i for i,pos in enumerate(Xlonlat) if 
                         (pos[0]>=118 and pos[0]<125 and # longitude boundaries
                          pos[1]>=-2.5 and pos[1]<2.5)] # latitude boundaries
    label[nort_sulawesi_idx] = sulawesi_label
    
    return label

def get_cluster_id(clf, lon, lat):
    lon_lat = np.array([lon, lat]).reshape(1,-1)
    cluster_id = clf.predict(lon_lat)
    return cluster_id

def equipments_segmentation(x):              
    if x.get('industry_sector')=='CON' and x.get('unit_model') in ['PC200-8', 'PC200-8M0']:
        segment = "{0}_{1}_{2}".format(x.get('unit_model'), x.get("industry_sector"), x.get('geo_cluster'))
    elif x.get('industry_sector')=='OTH' and x.get('unit_model') in ['PC200-8', 'PC200-8M0']:
        segment = "{0}_{1}_{2}".format(x.get('unit_model'), "CON", x.get('geo_cluster'))
    elif x.get('industry_sector') in ['FOR', 'AGR', 'OTH'] and x.get('unit_model') in ['PC400-8', 'PC300-8']:
        segment = "{0}_{1}".format(x.get('unit_model'), "AGR")
    elif x.get('industry_sector')!='MNG' and x.get('unit_model')=='D155A-6':
        segment = "{0}_{1}".format(x.get('unit_model'), "CON")
    else:
        segment = "{0}_{1}".format(x.get('unit_model'), x.get("industry_sector"))
    return segment