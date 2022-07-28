from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import datetime
from sklearn.metrics import silhouette_score
from django.http import JsonResponse
from django.db import connection

def getPoints(date_start=None, date_end=None):
    data_air_df = pd.DataFrame(columns=['deviceid', 'latitude', 'longitude', 'nama', 'average_distance', 'average_temp'])
    with connection.cursor() as cur:
        if (date_start is None) or (date_end is None):
            cur.execute("""
            SELECT 
                ds.sungai_deviceid,
                ds.sungai_latitude,
                ds.sungai_longtitude,
                ds.sungai_nama,
                da.average_distance,
                da.average_temp
            FROM master_sungai ds
            JOIN 
            (SELECT 
                deviceid, 
                ROUND(AVG(distance), 2) as average_distance,
                ROUND(AVG(temp), 2) as average_temp
            FROM (
                SELECT 
                    id, 
                    deviceid, 
                    date, 
                    temp, 
                    distance, 
                    @deviceid_rank:=IF(@current_deviceid = deviceid, @deviceid_rank + 1, 1) AS devicerank, 
                    @current_deviceid:=deviceid
                FROM (
                    SELECT * 
                    FROM data_airdb
                    ORDER BY 
                        deviceid ASC, id DESC
                ) data_air_sorted
                JOIN (SELECT @current_deviceid:=NULL, @deviceid_rank:=0) AS vars
                ORDER BY deviceid, id DESC
            ) data_ranked
            WHERE devicerank <= 100
            GROUP BY deviceid) da
            ON da.deviceid = ds.sungai_deviceid
            """)
            for r in cur.fetchall():
                data_air_df.loc[len(data_air_df)] = [r[0], r[1], r[2], r[3], r[4], r[5]]
        else:
            cur.execute("""
            SELECT 
                ds.sungai_deviceid,
                ds.sungai_latitude,
                ds.sungai_longtitude,
                ds.sungai_nama,
                da.average_distance,
                da.average_temp
            FROM master_sungai ds
            JOIN 
            (SELECT 
                deviceid, 
                ROUND(AVG(distance), 2) as average_distance,
                ROUND(AVG(temp), 2) as average_temp
            FROM (
                SELECT 
                    id, 
                    deviceid, 
                    date, 
                    temp, 
                    distance, 
                    @deviceid_rank:=IF(@current_deviceid = deviceid, @deviceid_rank + 1, 1) AS devicerank, 
                    @current_deviceid:=deviceid
                FROM (
                    SELECT * 
                    FROM data_airdb
                    WHERE date BETWEEN %s AND %s
                    ORDER BY deviceid ASC, id DESC
                ) data_air_sorted
                JOIN (SELECT @current_deviceid:=NULL, @deviceid_rank:=0) AS vars
                ORDER BY deviceid, id DESC
            ) data_ranked
            GROUP BY deviceid) da
            ON da.deviceid = ds.sungai_deviceid
            """, (date_start, date_end,))
            for r in cur.fetchall():
                data_air_df.loc[len(data_air_df)] = [r[0], r[1], r[2], r[3], r[4], r[5]]
    return data_air_df

# Create your views here.
def index(request):
    date_start = request.GET.get('date_start', None)
    date_end = request.GET.get('date_end', None)
    data_air_df = getPoints(date_start=date_start, date_end=date_end)
    data_air_x = data_air_df.iloc[:,[4,5]].to_numpy()
    silhoutte_score_list = []
    for i in range(2, 11):
        clusterer = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        clusterer.fit(data_air_x)
        silhoutte_score_list.append([silhouette_score(data_air_x, clusterer.labels_), i])
    silhouette_score_only_list = [x[0] for x in silhoutte_score_list]
    optimal_k_index = silhouette_score_only_list.index(max(silhouette_score_only_list))
    optimal_k = silhoutte_score_list[optimal_k_index][1]
    kmeans = KMeans(n_clusters=optimal_k)
    pred_y = kmeans.fit_predict(data_air_x)
    data_air_df['label'] = kmeans.labels_
    data_air_df = data_air_df.astype({'average_distance': 'float64'})
    result_list = []
    for cluster_label in range(0, optimal_k):
        result_list.append(data_air_df[data_air_df.label==cluster_label].sort_values(by=['deviceid']).to_dict(orient='records'))
    return JsonResponse(result_list, safe=False)