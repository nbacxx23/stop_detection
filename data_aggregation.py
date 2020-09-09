from __future__ import division
import pandas as pd
import numpy as np
import data_extraction
import mpl_toolkits.basemap.pyproj as pyproj
import matplotlib.pyplot as plt
import folium
import operator
import json
import os
import hadoopy
import math
import subprocess
import sys
import datetime

"""
Read typedbytes sequence file from HDFS and form proper dataset
    
Agrs:
    filen_ame (string(path)): file name which to be read
Returns:
    line (dataframe): dataframe with the information needed
        
"""
def extractInfo(file_name):
    line_raw = dict(hadoopy.readtb(file_name))
    line_raw = pd.DataFrame(line_raw.values(),columns=['DATE','TIME','LINE','BUS_NUM','X_COORDINATE','Y_COORDINATE'])
    line = data_extraction.getCoord(line_raw)
    return line

"""
Aggregate desiganted data(by month) to form the useful dataset. The function gets rid of the abnormal trajectories and calculates the
speed of each gps report according to the time sequence of one bus
    
Agrs:
    num_line (int): the line that is studied
    terminus1 (tuple): Abrev of the origin stop 
    terminus2 (tuple): Abrev of the terminus
    threshold (int): the range that decides the area of terminus (gps report beyond this range is considered as out of terminus area)
    *dirs (tuple): all the directories of the designated datasets
Returns:
    df (dataframe): proper dataframe containing all the information needed
        
"""
def aggregateData(num_line,terminus1,terminus2,threshold,*dirs):
    print 'aller:{}-{} \nretour:{}-{}'.format(terminus1,terminus2,terminus2,terminus1)
    ter_coor1,ter_coor2 = data_extraction.getTerminusCoor(num_line,terminus1,terminus2)
    source_gps_dir = 'hdfs://BigDataPOC:8020/datalab/exp_b02/data/gps_data'
    data_aller = []
    data_retour = []
    for dir_name in dirs:
        print dir_name
        path = os.path.join(source_gps_dir,str(num_line),str(dir_name))
        if hadoopy.isdir(path):
            cmd = 'hdfs dfs -du %s'%(path)
            p = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            frames_aller = []
            frames_retour = []
            for file_name in p.stdout.readlines():
                line_daily = extractInfo(file_name.split()[1])
                line_daily_aller,line_daily_retour = data_extraction.generateDailyData(line_daily,(ter_coor1,ter_coor2),threshold,terminus1,terminus2)
                frames_aller.append(line_daily_aller)
                frames_retour.append(line_daily_retour)
            data_aller.append(pd.concat(frames_aller).reset_index().drop('index',axis=1))
            data_retour.append(pd.concat(frames_retour).reset_index().drop('index',axis=1))
        else:
            if hadoopy.exists(dir_name):
                line_daily = extractInfo(dir_name)
                line_daily_aller,line_daily_retour = data_extraction.generateDailyData(line_daily,(ter_coor1,ter_coor2),threshold,terminus1,terminus2)
                data_aller.append(line_daily_aller)
                data_retour.append(line_daily_retour)
            else:
                print "there are paths in args which are not directories"
                sys.exit(1)
    data_aller = pd.concat(data_aller).reset_index().drop('index',axis=1)
    data_retour = pd.concat(data_retour).reset_index().drop('index',axis=1)
    cols = ['DATE', 'TIME', 'LINE', 'BUS_NUM', 'X_COORDINATE','Y_COORDINATE','LONGITUDE','LATITUDE','SPEED']
    data_aller = data_aller[cols]
    data_retour = data_retour[cols]
    return data_aller,data_retour

def addInformation(dataframe):
    df = dataframe.copy()
    for i in np.arange(len(df)):
        weekday = datetime.datetime.strptime(df.iloc[i].DATE,'%d/%m/%y').strftime('%A')
        hour = int(datetime.datetime.strptime(df.iloc[i].TIME,'%H:%M:%S').strftime('%H'))
        df.set_value(i, 'WEEKDAY', options_weekdays[weekday])
        df.set_value(i, 'TIME_RANGE', time_range(hour))
    return df

def time_range(hour):
    if 6<=hour and hour<8:
        return 1
    if 8<=hour and hour<10:
        return 2
    if 10<=hour and hour<12:
        return 3
    if 12<=hour and hour<14:
        return 4
    if 14<=hour and hour<16:
        return 5
    if 16<=hour and hour<18:
        return 6
    if 18<=hour and hour<20:
        return 7
    if 20<=hour and hour<22:
        return 8
    if 22<=hour and hour<24:
        return 9

options_weekdays = {'Monday' : 1,
            'Tuesday' : 2,
            'Wednesday' : 3,
            'Thursday' : 4,
            'Friday' : 5,
            'Saturday' : 6,
            'Sunday' : 7,
}