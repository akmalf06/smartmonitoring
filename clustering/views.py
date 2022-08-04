from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.metrics import silhouette_score
from django.http import HttpResponse, JsonResponse
from django.db import connection
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os

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

def prepDataRNN(datain, time_step):
    # 1. y-array  
    # Ambil semua index yang akan dijadikan label
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
    y_tmp = datain[y_indices]
    
    # 2. X-array  
    # Jumlah baris x harus sama dengan jumlah baris pada y atau label
    rows_X = len(y_tmp)
    # Ambil semua nilai X
    X_tmp = datain[range(time_step*rows_X)]
    # Reshape ke dalam bentuk sesuai time_step
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, 1))
    return X_tmp, y_tmp

def getPointsRNN(deviceid, date_start=None, date_end=None):
    data_air_df = pd.DataFrame(columns=['id', 'distance', 'temp', 'date'])
    with connection.cursor() as cur:
        if (date_start is None) or (date_end is None):
            cur.execute("""
            SELECT 
                da.id,
                da.distance,
                da.temp,
                da.date
            FROM master_sungai ds
            JOIN 
            (SELECT 
                id,
                deviceid, 
                distance,
                temp,
                date
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
                    ORDER BY id DESC
                ) data_air_sorted
                JOIN (SELECT @current_deviceid:=NULL, @deviceid_rank:=0) AS vars
                ORDER BY id DESC
            ) data_ranked
            WHERE devicerank <= 100) da
            ON da.deviceid = ds.sungai_deviceid
            WHERE da.deviceid = %s;
            """, (deviceid,))
            for r in cur.fetchall():
                data_air_df.loc[len(data_air_df)] = [r[0], r[1], r[2], r[3]]
        else:
            cur.execute("""
            SELECT 
                da.id,
                da.distance,
                da.temp,
                da.date
            FROM master_sungai ds
            JOIN 
            (SELECT 
                id,
                deviceid, 
                distance,
                temp,
                date
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
                    ORDER BY id DESC
                ) data_air_sorted
                JOIN (SELECT @current_deviceid:=NULL, @deviceid_rank:=0) AS vars
                ORDER BY id DESC
            ) data_ranked ) da
            ON da.deviceid = ds.sungai_deviceid
            WHERE da.deviceid = %s;
            """, (date_start, date_end, deviceid,))
            for r in cur.fetchall():
                data_air_df.loc[len(data_air_df)] = [r[0], r[1], r[2], r[3]]
    return data_air_df

def getNSecondPrediction(df_air, model, second=300):
    n_iteration = second//2
    last_7_df = df_air.tail(7).copy()
    x_pred_final = []
    y_pred_final = []
    for i in range(n_iteration):
        X=last_7_df.tail(7)[['distance']].copy()
        scaler = MinMaxScaler()
        X_scaled=scaler.fit_transform(X)
        X_ready = X_scaled[range(7)]
        X_ready = np.reshape(X_ready, (1, 7, 1))
        pred_input = model.predict(X_ready, verbose=False)
        x_pred_final += scaler.inverse_transform(pred_input).flatten().tolist()
        last_7_df.loc[len(last_7_df)] = [i, x_pred_final[-1]]
        y_pred_final.append(i)
    return x_pred_final, y_pred_final

# Create your views here.
def index(request):
    return HttpResponse("<html><body>SMART MITIGATION API</body></html>")

def cluster(request):
    date_start = request.GET.get('date_start', None)
    date_end = request.GET.get('date_end', None)
    data_air_df = getPoints(date_start=date_start, date_end=date_end)
    if data_air_df.empty:
        return JsonResponse([], safe=False)
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

def rnnprediction(request):
    deviceid = request.GET.get('deviceid', None)
    if(deviceid is None):
        return JsonResponse({
            'success': 'false',
            'message': 'Deviceid tidak diberikan'    
        }, safe=False)
    date_start = request.GET.get('date_start', None)
    date_end = request.GET.get('date_end', None)
    second = request.GET.get('second', 300)
    second = int(second)
    df = getPointsRNN(deviceid, date_start, date_end)
    if(df.shape[0] < 7):
        return JsonResponse({
            'success': 'false',
            'message': 'Data setidaknya berjumlah 7 record'    
        }, safe=False)
    data_air_df=df[['date', 'distance']].copy()
    data_air_input = data_air_df.to_numpy().tolist()
    X_raw_input=data_air_df[['distance']]
    scaler = MinMaxScaler()
    X_scaled=scaler.fit_transform(X_raw_input)
    X_input, y_input = prepDataRNN(X_scaled, 7)
    modelPath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RNNModel'))
    model = load_model(modelPath)
    pred_input = model.predict(X_input)
    mse = mean_squared_error(y_input, pred_input)
    y_input_final = scaler.inverse_transform(y_input).flatten()
    x_input_final = np.array(range(0,len(y_input_final)))
    test_data = np.column_stack((x_input_final, y_input_final)).tolist() 
    y_pred_final = scaler.inverse_transform(pred_input).flatten()
    x_pred_final = np.array(range(0,len(y_pred_final)))
    prediction = np.column_stack((x_pred_final, y_pred_final)).tolist()
    x_prediction, y_prediction = getNSecondPrediction(data_air_df, model, second)
    prediction_second = list(zip(x_prediction, y_prediction))
    return JsonResponse({ 
        "time_step": 7, 
        "mse": mse,
        "input_data": data_air_input,
        "test_data": test_data,
        "prediction": prediction,
        "prediction_second": prediction_second
    }, safe=False)

def pointByClass(request):
    date_start = request.GET.get('date_start', None)
    date_end = request.GET.get('date_end', None)
    data_air_df = getPoints(date_start=date_start, date_end=date_end)
    if data_air_df.empty:
        return JsonResponse({}, safe=False)
    result = {}
    temp_df = data_air_df[data_air_df['average_distance']<=5].copy()
    bottom_limit = 0
    upper_limit = 5
    while upper_limit <= 35:
        key = ''
        if upper_limit == 5:
            temp_df = data_air_df[data_air_df['average_distance'] <= 5].copy()
            key = 'le5'
        if upper_limit == 35:
            temp_df = data_air_df[data_air_df['average_distance'] > 30].copy()
            key = 'm30'
        if (upper_limit > 5) and (upper_limit < 35):
            temp_df = data_air_df[(data_air_df['average_distance'] > bottom_limit) & (data_air_df['average_distance'] <= upper_limit)].copy()
            key = "m{}le{}".format(str(bottom_limit), str(upper_limit))
        n = temp_df.shape[0]
        name = None
        for index, row in temp_df.iterrows():
            if name is None:
                name = str(row['deviceid'])
            else:
                name = name + ', ' + str(row['deviceid'])
        distance_mean = temp_df['average_distance'].mean() if not math.isnan(temp_df['average_distance'].mean()) else 0
        temp_mean = temp_df['average_temp'].mean() if not math.isnan(temp_df['average_temp'].mean()) else 0
        result[key] = {
            'total': n,
            'distance_mean': distance_mean,
            'temp_mean': temp_mean,
            'deviceids': name
        }
        bottom_limit += 5
        upper_limit += 5
    return JsonResponse(result)