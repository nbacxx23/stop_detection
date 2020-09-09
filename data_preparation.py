import numpy as np
import pandas as pd
import csv
import sys
from dateutil.parser import parse
import hadoopy
import os

def extractGPS(file_name):
    coordinate = []
    with open(file_name,'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter=';',quotechar='|')
        for row in reader:
            if 'Xrec' in row:
                coordinate.append(row)
    csvfile.close()
    length = len(sorted(coordinate,key=len, reverse=True)[0])
    gps_data=np.array([xi+[None]*(length-len(xi)) for xi in coordinate])
    gps_data = pd.DataFrame(gps_data)
    gps_data = gps_data.iloc[:,[0,12,2,7,14,16]]
    gps_data.columns = ['Date','Time','Line','Bus_num','X_coordinate','Y_coordinate']
    return gps_data

def dataClean(file_name):
    fi = open(file_name, 'rb')
    data = fi.read()
    fi.close()
    fo = open(file_name, 'wb') 
    fo.write(data.replace('\x00', ''))
    fo.close()
    
def uniformTimeFormat(dataframe):
    df = dataframe.copy()
    for i in np.arange(len(df)):
        df.set_value(i,'Time',"%02d:%02d:%02d"%(int(df['Time'][i].split(':')[0]),
                       int(df['Time'][i].split(':')[1]),int(df['Time'][i].split(':')[2])))
    return df

def readHDFS(path):
    data_raw = dict(hadoopy.readtb(path))
    coordinate = []
    for row in data_raw.itervalues():
        if 'Xrec' in row:
            coordinate.append(row)
    if coordinate == []:
        return pd.DataFrame()
    length = len(sorted(coordinate,key=len, reverse=True)[0])
    coordinate_list = [x.encode('UTF8').split(';') for x in coordinate]
    gps_data=np.array([xi+[None]*(length-len(xi)) for xi in coordinate_list])
    gps_data = pd.DataFrame(gps_data)
    gps_data = gps_data.iloc[:,[0,12,2,7,14,16]]
    gps_data.columns = ['Date','Time','Line','Bus_num','X_coordinate','Y_coordinate']
    return gps_data

def getGpsData(source,destination):
    gps_data = readHDFS(source)
    if gps_data.empty == False:
        gps_data = uniformTimeFormat(gps_data)
        data_trans = gps_data.T.to_dict('list')
        tuples = [ item for item in data_trans.iteritems()]
        hadoopy.writetb(destination,tuples)
        
def extractUsefulData(num_line,start_date,end_date):
    year = str(start_date)[:4]
    month = str(start_date)[4:6]
    start_day = str(start_date)[-2:]
    end_day = str(end_date)[-2:]
    home_dir_source = 'hdfs://BigDataPOC:8020/datalab/exp_vsb/inputData'
    home_dir_des = 'hdfs://BigDataPOC:8020/datalab/exp_b02/data/gps_data'
    for i in np.arange(int(start_day),int(end_day)+1):
        if i<10:
            date = '0'+ str(i)
        else:
            date = str(i)
        file_source = 'loc_bus_'+ str(start_date)[:6] +date+'_'+str(num_line)+'.csv' 
        source = os.path.join(home_dir_source,file_source)
        home_dir_des_line = os.path.join(home_dir_des,str(num_line))
        home_dir_des_month = os.path.join(home_dir_des_line,str(start_date)[:6])
        if not os.path.exists(home_dir_des_line):
            try:
                os.mkdir(os.path.dirname(home_dir_des_line))
            except OSError:
                pass
            if not os.path.exists(home_dir_des_month):
                try:
                    os.mkdir(os.path.dirname(home_dir_des_month))
                except OSError:
                    pass
        if not os.path.exists(home_dir_des_month):
                try:
                    os.mkdir(os.path.dirname(home_dir_des_month))
                except OSError:
                    pass
        file_des = 'bus_gps_'+ str(start_date)[:6] +date+'_'+str(num_line)+'.csv' 
        destination = os.path.join(home_dir_des_month,file_des)
        if hadoopy.exists(destination):
            hadoopy.rmr(destination)
        getGpsData(source,destination)
        print 'it is finished:'+file_des
        
if __name__ == '__main__':    
    if len(sys.argv) != 4:
            print 'Invalid number of arguments passed.'
            print 'Correct usage: python data_preparation.py num_line start_date end_date'
    else: 
        extractUsefulData(sys.argv[1],sys.argv[2],sys.argv[3])