import numpy as np
import pandas as pd
import data_extraction
from sklearn import preprocessing
import line_model

def filter_final_stop(features_c,features,threshold,target):
    group = {}
    count = 1
    group[count]=[features_c.keys()[0]]
    temp = features_c.copy()
    del temp[features_c.keys()[0]]
    while len(temp)!=0:
        flag=0
        for key in temp.keys():
            for j in xrange(len(group[count])):
                if data_extraction.mesureDistance(key,group[count][j])<threshold:
                    group[count].append(key)
                    del temp[key]
                    flag=1
                    break
        if flag==0:
            count+=1
            group[count] = [temp.keys()[0]]
            del temp[temp.keys()[0]]
    result = []
    for k in group.itervalues():
        temp = []
        for c in k:
            temp.append(features[c][target])
        index = temp.index(max(temp))
        result.append(k[index])
    return result

def clean_density_value(density):
    #value = (np.sqrt(speed_var)+1)*(1/density)*(1+(speed))
    value = 1/density
    #value = (1+1/(speed_var+1))*(1/np.log(density+1))*(1+speed)
    #value = (1+speed_var)*(1/np.log(density+1))*(1+speed)
    return value

def filter_outlier(speed,speed_var):
    value = (np.sqrt(speed_var)+1)*(1+(speed))
    return value

def seperator_cluster(features_train,labels,target_features):
    positive = np.argwhere(labels==1)
    positive_features = np.array(features_train.values())[positive,target_features]
    #positive_features = preprocessing.scale(positive_features)
    s  = np.arange(len(features_train))
    s = np.delete(s,positive)
    negative_features = np.array(features_train.values())[s.reshape(len(s),1),target_features]
    #negative_features = preprocessing.scale(negative_features)
    return positive_features,negative_features

def get_label(result,line,features):
    cluster_stop = []
    for i in xrange(len(result)):
        x = result[i][0]
        y = result[i][1]
        cluster = line_model.findCluster(line.cluster,line.cluster_kdtree,(x,y))
        cluster_stop.append(cluster)
    labels = []
    for cluster in features:
        if cluster in cluster_stop:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels) 
    return labels

def threshold_learning(positive_cluster):
    value_list=[]
    for f in positive_cluster:
        value_list.append(clean_density_value(f[2]))
    value_speed_var = []
    for f_v in positive_cluster:
        value_speed_var.append(filter_outlier(f_v[1],f_v[3]))
    threshold_den = sorted(value_list,reverse=True)[0]
    threshold_vr = sorted(value_speed_var,reverse=True)[0]
    return threshold_den,threshold_vr

def learning_stop(features,threshold_den,threshold_vr):                          
    stop_list = []
    for i in np.arange(len(features)):
        value = clean_density_value(features.values()[i][13])
        if value<threshold_den:
            stop_list.append(i)
    stop_list2 = []
    for j in stop_list:
        value = filter_outlier(features.values()[j][11],features.values()[j][14])
        if value<threshold_vr:
            stop_list2.append(j)
    return stop_list2