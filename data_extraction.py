#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:51:13 2017

@author: xc734879
"""

import numpy as np
import pandas as pd
import datetime
import mpl_toolkits.basemap.pyproj as pyproj
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pandas.tseries.offsets import Second
import sys
import os
import hadoopy
import math

"""
Add columns(longitude and latitude) to the original gps dataframe
    
Agrs:
    dataframe (dataframe): original gps dataframe
Returns:
    df (dataframe): dataframe with 'LONGITUDE' and 'LATITUDE'
        
"""
def getCoord(dataframe):
    df = dataframe.copy()
    WGS84 = pyproj.Proj(init='epsg:4326')
    Lambert2 = pyproj.Proj(init='epsg:27572')
    for i in np.arange(len(df)):
        x = df.ix[i].X_COORDINATE
        y = df.ix[i].Y_COORDINATE
        lons,lats = pyproj.transform(Lambert2,WGS84,x,y)
        df.set_value(i,'LONGITUDE',lons)
        df.set_value(i,'LATITUDE',lats)
    return df

"""
Extract terminus' coordinates according to the line and abrevs. Reference file is stored on HDFS.
    
Agrs:
    num_line (int): the line that is studied
    terminus1 (tuple): Abrev of the origin stop
    terminus2 (tuple): Abrev of the terminus
Returns:
    coor1,coor2 (int,int): coordinates of origin stop/terminus 
        
"""
def getTerminusCoor(num_line,terminus1,terminus2):
    home_dir = 'hdfs://BigDataPOC:8020/datalab/exp_b02/data/gps_data'
    file_name = 'Stop_Ref_'+str(num_line)+'.csv'
    path = os.path.join(home_dir,str(num_line),file_name)
    ref_r = dict(hadoopy.readtb(path))
    ref = pd.DataFrame([x.encode('UTF8').split(';') for x in ref_r.values()],columns=str(ref_r[0]).split(';'))
    ter1 = []
    ter2 = []
    for i in np.arange(len(ref)):
        if ref.iloc[i].Abrev_MP==terminus1:
            ter1.append((int(ref.iloc[i].X),int(ref.iloc[i].Y)))
        elif ref.iloc[i].Abrev_MP ==terminus2:
            ter2.append((int(ref.iloc[i].X),int(ref.iloc[i].Y)))
    coor1 = (math.floor((ter1[0][0]+ter1[1][0])/2),math.floor((ter1[0][1]+ter1[1][1])/2))
    coor2 = (math.floor((ter2[0][0]+ter2[1][0])/2),math.floor((ter2[0][1]+ter2[1][1])/2))
    return coor1,coor2


"""
Detect whether a point is within the range of terminus
    
Agrs:
    x,y (tuple): coordinate of the point
    x_o,y_o (tuple): coordinate of terminus1
    x_t,y_t (tuple): coordinate of terminus2
    threshold (int): the range of domain of the terminus
Returns:
    (Boolean,Boolean): signal whether the point is within the range of terminus 
        
"""
# (x_o,y_o) -> hôtel de ville  (601048,2428726)
# (x_t,y_t) -> saint-cloud  (591609,2427194)
def terminusDetect((x,y),(x_o,y_o),(x_t,y_t),threshold):
    distance_o = np.sqrt(pow((x-x_o),2)+pow((y-y_o),2))
    distance_t = np.sqrt(pow((x-x_t),2)+pow((y-y_t),2))
    if distance_o<=threshold:
        return True,False
    elif distance_t <=threshold:
        return False,True
    else:
        return False,False 

"""
Extract abnormal trajectories
    
Agrs:
    Terminus (list): gps reports within the range of all the terminus
Returns:
    abnormal (list): abnormal trajectories in a certain period
    
Definition of the situation abnormal - a complete trajectory is considered as a trajectory with different terminus 
reported in more than 300 seconds. On contrary, the consecutive reports do not fit for the condition above are considered abnormal         
"""
def getTraject(Terminus):
    traject = []
    abnormal = []
    time = lambda terminus:terminus[1]
    index = lambda terminus:terminus[3]
    stopType = lambda terminus:terminus[4]
    for i in np.arange(len(Terminus)-1):
        if stopType(Terminus[i+1])!=stopType(Terminus[i]):
            traject.append((index(Terminus[i]),index(Terminus[i+1]),stopType(Terminus[i])))
    return traject

def extractCleanTraces(bus_num,(terminus1,terminus2),threshold,ter_nom1,ter_nom2):
    # 'S' -> hotel de ville
    # 'T' -> saint-cloud
    Terminus = []
    for i in np.arange(len(bus_num)):
        isOrignal,isTerminus = terminusDetect((float(bus_num.iloc[i].X_COORDINATE),float(bus_num.iloc[i].Y_COORDINATE)),
                                          terminus1,terminus2,threshold)
        if isOrignal:
            Terminus.append((bus_num.iloc[i].BUS_NUM,bus_num.iloc[i].TIME,
                         (bus_num.iloc[i].X_COORDINATE,bus_num.iloc[i].Y_COORDINATE),i,ter_nom1))
        elif isTerminus:
            Terminus.append((bus_num.iloc[i].BUS_NUM,bus_num.iloc[i].TIME,
                         (bus_num.iloc[i].X_COORDINATE,bus_num.iloc[i].Y_COORDINATE),i,ter_nom2))
    
    if Terminus == []:
        return pd.DataFrame(),pd.DataFrame()
    traject = getTraject(Terminus)
    if traject == []:
        return pd.DataFrame(),pd.DataFrame()
    #begin,end = Terminus[0][3],Terminus[-1][3]
    #bus_num.drop(bus_num.index[end+1:],inplace=True)
    #bus_num.drop(bus_num.index[:begin-1],inplace=True)
    '''
    if ab == []:
        bus_num = bus_num.drop_duplicates()
        return bus_num
    else:
    '''
    data_aller = []
    data_retour = []
    for points in traject:
        if points[2] == ter_nom1:
            data_aller.append(bus_num.iloc[points[0]:points[1]+1])
        else:
            data_retour.append(bus_num.iloc[points[0]:points[1]+1])
    if data_aller!=[]: 
        clean_data_aller = pd.concat(data_aller).reset_index().drop('index',axis=1)
    else:
        clean_data_aller = pd.DataFrame()
    if data_retour!=[]: 
        clean_data_retour = pd.concat(data_retour).reset_index().drop('index',axis=1)
    else:
        clean_data_retour = pd.DataFrame()
    return clean_data_aller,clean_data_retour

def generateDailyData(line,(terminus1,terminus2),threshold,abbr_ter1,abbr_ter2):
    bus_num = np.unique(line['BUS_NUM'].values)
    frames_aller = []
    frames_retour = []
    for bus in bus_num:
        bus_target = line[line['BUS_NUM']== bus]
        bus_target = bus_target.reset_index().drop('index',axis=1)
        bus_target = bus_target.sort_values(['TIME'],ascending=[True])
        #print bus,(terminus1,terminus2),threshold,abbr_ter1,abbr_ter2
        clean_trace_aller,clean_trace_retour = extractCleanTraces(bus_target,(terminus1,terminus2),threshold,abbr_ter1,abbr_ter2)
        if len(clean_trace_aller)!=0:
            clean_trace_aller = calculateSpeed(clean_trace_aller,20)
            frames_aller.append(clean_trace_aller)
        if len(clean_trace_retour)!=0:
            clean_trace_retour = calculateSpeed(clean_trace_retour,20)
            frames_retour.append(clean_trace_retour)
    daily_data_aller = pd.concat(frames_aller).reset_index().drop('index',axis=1)
    daily_data_retour = pd.concat(frames_retour).reset_index().drop('index',axis=1)
    return daily_data_aller,daily_data_retour

def mesureDistance((x1,y1),(x2,y2)):
    distance = np.sqrt(pow((x1-x2),2)+pow((y1-y2),2))
    return distance

def calculateSpeed(clean_trace,limit):
    trace = clean_trace.copy()
    abnorm = []
    speed_limit = limit
    #hyper_speed = []
    # delete repeated reports
    for i in np.arange(1,len(trace)):
        period = parse(trace.iloc[i].TIME)-parse(trace.iloc[i-1].TIME)
        period = period.total_seconds()
        if period == 0:
            abnorm.append(i-1)
    trace = trace.drop(trace.index[abnorm])
    trace = trace.reset_index().drop('index',axis=1)
    # calculate the speed based on the precedent and the next gps report
    for i in xrange(1,len(trace)-1):
        period = parse(trace.iloc[i+1].TIME)-parse(trace.iloc[i-1].TIME)
        period = period.total_seconds()
        x1 = trace.iloc[i+1].X_COORDINATE
        y1 = trace.iloc[i+1].Y_COORDINATE
        x2 = trace.iloc[i-1].X_COORDINATE
        y2 = trace.iloc[i-1].Y_COORDINATE
        distance = mesureDistance((int(x1),int(y1)),(int(x2),int(y2)))
        speed = distance/period
        speed = round(speed,3)
        if period != 0 and speed >= 0 and speed<=speed_limit:
            trace.set_value(i,'SPEED',speed)
    row_nan = np.argwhere(np.array(pd.isnull(trace)))
    list_nan = [i[0] for i in row_nan]
    list_nan.extend([0,-1])
    trace = trace.drop(trace.index[list_nan])
    trace = trace.reset_index().drop('index',axis=1)
    '''
    for i in xrange(len(trace)):
        speed = trace.iloc[i].SPEED
        k = 0
        while speed > speed_limit or speed<0:
            if i-2-k<0:
                hyper_speed.append(i)
                break
            period = parse(trace.iloc[i].TIME)-parse(trace.iloc[i-2-k].TIME)
            period = period.total_seconds()
            if period <= 0 :
                k+=1
                continue
            x1 = trace.iloc[i].X_COORDINATE
            y1 = trace.iloc[i].Y_COORDINATE
            x2 = trace.iloc[i-2-k].X_COORDINATE
            y2 = trace.iloc[i-2-k].Y_COORDINATE
            speed_c = mesureDistance((int(x1),int(y1)),(int(x2),int(y2)))/period
            if speed_c <= speed_limit and speed>=0:
                speed = speed_c
                speed = round(speed,3)
                trace.set_value(i,'SPEED',speed)
                for j in xrange(i-1,i-2-k,-1):
                    hyper_speed.append(j)
                break
            else:
                k+=1
    trace = trace.drop(trace.index[hyper_speed])
    trace = trace.reset_index().drop('index',axis=1)
    '''
    return trace

def set_seperate_density(dataframe,feature):
    df = dataframe.copy()
    feature_f = feature.copy()
    density_weekday = pd.DataFrame({'sum_weekdays':df.groupby(['cluster_set','WEEKDAY']).size()}).reset_index()
    density_timerange = pd.DataFrame({'sum_timerange':df.groupby(['cluster_set','TIME_RANGE']).size()}).reset_index()
    for i in density_weekday.iterrows():
        cluster,weekday,count= i[1].cluster_set,i[1].WEEKDAY,i[1].sum_weekdays
        feature_f[cluster][int(weekday)+14] = count
    for i in density_timerange.iterrows():
        cluster,timerange,count= i[1].cluster_set,i[1].TIME_RANGE,i[1].sum_timerange
        feature_f[cluster][int(timerange)+21] = count
    return feature_f

if __name__ == '__main__':    
    if len(sys.argv) == 3:
        print 'Invalid number of arguments passed.'
        print 'Correct usage: python segmentation.py input_file output_file'
    else:
        gps_file_name = '/appli/bgd/exp_b02/data/bus/72/bus_gps_20160302_72_raw.csv'
        fields=['DATE','TIME','BUS_NUM','X_COORDINATE','Y_COORDINATE']
        line_raw = pd.read_csv(gps_file_name,usecols=fields,sep=';')
        line = transformCoord(line_raw)
        daily_data = generateDailyData(line,(601048,2428726),(591609,2427194),100)
        daily_data.to_csv('/appli/bgd/exp_b02/data/clean_data_gps_20160302_72.csv',index=False)