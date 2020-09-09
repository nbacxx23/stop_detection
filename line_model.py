#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:39:01 2017

@author: xc734879
"""

from __future__ import division
import numpy as np
import pandas as pd
import datetime
import mpl_toolkits.basemap.pyproj as pyproj
import matplotlib.pyplot as plt
import folium
import math
from sklearn.neighbors import KDTree
import operator
import sys
from data_extraction import mesureDistance
import segmentation_interpolation

class Line:
    """
    Initialize necessary variables 
    
    Agrs:
        num_line (int) : number of the line   
        d_mini (int): unit distance of mini cluster
        d_cluster (int): unit distance of cluster set
    Returns:
        Object of a specific Line class
    
    Variables-
    self.min_x: 
    self.max_x:
    self.min_y:
    self.max_y:
    """
    def __init__(self,num_line,terminus1,terminus2,d_mini,d_cluster,threshold_mini,threshold_cluster,direction):
        self.d_mini = d_mini
        self.d_cluster = d_cluster
        self.num_line = num_line
        self.terminus1 = terminus1
        self.terminus2 = terminus2
        self.threshold_mini = threshold_mini
        self.threshold_cluster = threshold_cluster
        self.construct_model(direction)
    
    """
    Construct the specific line model
    
    Methods -
    get_bounds(): set the boundaries of the line model based on the referential segments
    generate_map_grid(): generate mini_cluster grid based on the referential segments
    generate_cluster_set(): generate cluster set based on the mini_cluster grid
    relation_cluster(): establish the connection beween mini_cluster and cluster set
    """
    def construct_model(self,direction):
        seg = segmentation_interpolation.getSegment(self.num_line,direction)
        self.get_bounds(pd.concat(seg,axis=0).reset_index().drop('index',axis=1))
        ref_f = segmentation_interpolation.itinerary_extraction(seg)
        self.generate_map_grid(ref_f)
        self.generate_cluster_set()
        self.relation_cluster()
    
    """
    Decide the boundary of the two axis
    """
    def get_bounds(self,seg):
        self.min_x = min(seg['X'].values)
        self.max_x = max(seg['X'].values)
        self.min_y = min(seg['Y'].values)
        self.max_y = max(seg['Y'].values)
        '''
        if self.min_x > min(seg2['X'].values):
            self.min_x = min(seg2['X'].values)
        if self.max_x < max(seg2['X'].values):
            self.max_x = max(seg2['X'].values)
        if self.min_y > min(seg2['Y'].values):
            self.min_y = min(seg2['Y'].values)
        if self.max_y < max(seg2['Y'].values):
            self.max_y = max(seg2['Y'].values)
        '''
        self.min_x = int(self.min_x) + 20
        self.max_x = int(self.max_x) + 20
        self.min_y = int(self.min_y) + 20
        self.max_y = int(self.max_y) + 20
        print self.min_x,self.max_x,self.min_y,self.max_y
    
    """
    Generate mini_cluster grid based on the referential segments
    
    Agrs:
        ref_f (set): the features represent the distributed density histograms
        
    self.map_grid: list of mini_cluster center coordinates
    self.grid_kdtree: kdtree representation of the map grid
    """
    def generate_map_grid(self,ref_f):
        range_x = math.ceil((self.max_x-self.min_x)/self.d_mini)
        range_y = math.ceil((self.max_y-self.min_y)/self.d_mini)
        print range_x,range_y
        self.map_grid = set()
        ref_f = list(ref_f)
        kdtree_ref = KDTree(ref_f)
        for y in np.arange(range_y):
            for x in np.arange(range_x):
                centroid = (self.min_x+(x+0.5)*self.d_mini,self.min_y+(y+0.5)*self.d_mini)
                coor = findCluster(ref_f,kdtree_ref,centroid)
                distance = mesureDistance(centroid,coor)
                if distance <= self.threshold_mini:
                    self.map_grid.add(centroid)
        self.map_grid = list(self.map_grid)
        self.grid_kdtree = KDTree(self.map_grid)
                
    def generate_cluster_set(self):
        range_set_x = math.ceil((self.max_x-self.min_x)/self.d_cluster)
        range_set_y = math.ceil((self.max_y-self.min_y)/self.d_cluster)
        print range_set_x,range_set_y
        cluster_raw = []
        self.cluster = set()
        for y in np.arange(range_set_y):
            for x in np.arange(range_set_x):
                centroid = (self.min_x+(x+0.5)*self.d_cluster,self.min_y+(y+0.5)*self.d_cluster)
                coor = findCluster(self.map_grid,self.grid_kdtree,centroid)
                distance = mesureDistance(centroid,coor)
                if distance <= self.threshold_cluster:
                    self.cluster.add(centroid)
        self.cluster = list(self.cluster)
        self.cluster_kdtree = KDTree(self.cluster)
                
    def relation_cluster(self):
        self.relation_cluster={}
        for point in self.map_grid:
            coor = findCluster(self.cluster,self.cluster_kdtree,(point[0],point[1]))
            self.relation_cluster[(point[0],point[1])] = coor
    
    """
    Perform statistic work (density of all the mini-cluster) and add corresponding mini/cluster for each gps report
    
    Agrs:
        dataframe (dataframe): dataset
    Returns:
        df (dataframe): dataframe with corresponding 'mini_cluster' center and 'cluster_set' center added
    
    """         
    def density_stat(self,dataFrame):
        df = dataFrame.copy()
        df['mini_cluster'] = pd.Series(zip(np.zeros(len(df)),np.zeros(len(df))),index=df.index)
        df['cluster_set'] = pd.Series(zip(np.zeros(len(df)),np.zeros(len(df))),index=df.index)
        self.mini_cluster_stat = {}
        drop_list = []
        for i in np.arange(len(df)):
            x = df.iloc[i].X_COORDINATE
            y = df.iloc[i].Y_COORDINATE
            coor = findCluster(self.map_grid,self.grid_kdtree,(x,y))
            distance = mesureDistance((x,y),coor)
            if distance <= self.threshold_mini:
                if not self.mini_cluster_stat.has_key(coor):
                    self.mini_cluster_stat[coor] = 1
                else: self.mini_cluster_stat[coor] = self.mini_cluster_stat[coor] + 1
                df.set_value(i,'mini_cluster',coor)
                df.set_value(i,'cluster_set',self.relation_cluster[coor])
            else:
                drop_list.append(i)
        df = df.drop(df.index[drop_list])
        df = df.reset_index().drop('index',axis=1)
        self.c_max = sorted(self.mini_cluster_stat.items(),key=operator.itemgetter(1),reverse=True)[2][1]
        return df
    
    """
    Calculates the speed and density features of each cluster and add them 
    into the already existing histogram features
    
    Agrs:
        histogram (narray): the features represent the distributed density histograms.
        dataframe (dataframe): data that is used to extract useful information
        calculate_speed (boolean) : True if speed needed to be calculated
        generate_feature (boolean) : True if new features need to be generated
    Returns:
        his (narray): the features added with speed characteristics and absolute density
        
    self.cluster_stat[0]: gps report numbers with speed < 2 m/s
    self.cluster_stat[1]: the ratio of low speed(<2 m/s)
    self.cluster_stat[2]: total accumulated speed in the cluster
    self.cluster_stat[3]: speed expectation of the cluster
    self.cluster_stat[4]: density of the cluster
    self.cluster_stat[5]: variance of speed of the cluster
    """
    def feature_his(self,factor,feature_num):
        feature = {}
        threshold_1 = factor*self.c_max
        threshold_2 = 2*factor*self.c_max
        threshold_3 = 3*factor*self.c_max
        threshold_4 = 4*factor*self.c_max
        threshold_5 = 5*factor*self.c_max
        threshold_6 = 6*factor*self.c_max
        threshold_7 = 7*factor*self.c_max
        threshold_8 = 8*factor*self.c_max
        threshold_9 = 9*factor*self.c_max
        for cluster in self.cluster:
            feature[cluster] = np.zeros(feature_num)
        for mini_cluster in self.mini_cluster_stat:
            value = self.mini_cluster_stat[mini_cluster]
            if value<=threshold_1:
                feature[self.relation_cluster[mini_cluster]][0]+=1
                continue
            elif threshold_1<value and value<=threshold_2:
                feature[self.relation_cluster[mini_cluster]][1]+=1
                continue
            elif threshold_2<value and value<=threshold_3:
                feature[self.relation_cluster[mini_cluster]][2]+=1
                continue
            elif threshold_3<value and value<=threshold_4:
                feature[self.relation_cluster[mini_cluster]][3]+=1
                continue
            elif threshold_4<value and value<=threshold_5:
                feature[self.relation_cluster[mini_cluster]][4]+=1
                continue
            elif threshold_5<value and value<=threshold_6:
                feature[self.relation_cluster[mini_cluster]][5]+=1
                continue
            elif threshold_6<value and value<=threshold_7:
                feature[self.relation_cluster[mini_cluster]][6]+=1
                continue
            elif threshold_7<value and value<=threshold_8:
                feature[self.relation_cluster[mini_cluster]][7]+=1
                continue
            elif threshold_8<value and value<=threshold_9:
                feature[self.relation_cluster[mini_cluster]][8]+=1
                continue
            elif value>threshold_9:
                feature[self.relation_cluster[mini_cluster]][9]+=1
        feature_n = {}
        s = np.array(feature.values()).sum(axis=1)
        index = [i for i,e in enumerate(s) if e!=0]
        for i in index:
            feature_n[feature.keys()[i]]=feature.values()[i]/sum(feature.values()[i])
        for cluster,features in feature_n.items():
            feature_n[cluster][12] = feature_n[cluster][0] + feature_n[cluster][1]
        return feature_n
    
    """
    Calculates the speed and density features of each cluster and add them 
    into the already existing histogram features
    
    Agrs:
        histogram (narray): the features represent the distributed density histograms.
        dataframe (dataframe): data that is used to extract useful information
        calculate_speed (boolean) : True if speed needed to be calculated
        generate_feature (boolean) : True if new features need to be generated
    Returns:
        his (narray): the features added with speed characteristics and absolute density
        
    self.cluster_stat[0]: gps report numbers with speed < 2 m/s
    self.cluster_stat[1]: the ratio of low speed(<2 m/s)
    self.cluster_stat[2]: total accumulated speed in the cluster
    self.cluster_stat[3]: speed expectation of the cluster
    self.cluster_stat[4]: density of the cluster
    self.cluster_stat[5]: variance of speed of the cluster
    """
    def feature_density_speed(self,dataframe,calculate_speed=False,generate_feature=False,histogram=None):
        df = dataframe.copy()
        if histogram!=None:
            his = histogram.copy()
        self.cluster_stat = {}
        cluster_speed = {}
        for cluster in histogram.iterkeys():
            self.cluster_stat[cluster] = np.zeros(6)
            cluster_speed[cluster] = []
        for mini_cluster,density in self.mini_cluster_stat.iteritems():
            self.cluster_stat[self.relation_cluster[mini_cluster]][4] +=density
        if calculate_speed:
            for i in np.arange(len(df)):
                speed = df.iloc[i].SPEED
                self.cluster_stat[df.iloc[i].cluster_set][2]+=speed
                cluster_speed[df.iloc[i].cluster_set].append(speed)
                if speed <= 2:
                    self.cluster_stat[df.iloc[i].cluster_set][0] += 1
            for cluster in self.cluster_stat.keys():
                self.cluster_stat[cluster][1] = self.cluster_stat[cluster][0]/self.cluster_stat[cluster][4]
                self.cluster_stat[cluster][3] = self.cluster_stat[cluster][2]/self.cluster_stat[cluster][4]
                self.cluster_stat[cluster][5] = np.var(cluster_speed[cluster])
        if generate_feature:
            for cluster in self.cluster_stat.keys():
                his[cluster][10] = self.cluster_stat[cluster][1]
                his[cluster][11] = self.cluster_stat[cluster][3]
                his[cluster][13] = self.cluster_stat[cluster][4]
                his[cluster][14] = self.cluster_stat[cluster][5]
            return his

"""
Searching for the corresponding cluster according to the coordinates provided 
    
Agrs:
    cluster (list): list of all the clusters of the line
    kd_tree (KDTree): with geo-structured information of all the clusters
    (x,y) (tuple): coordinate provided to search for the cluster
Returns:
    coor (tuple): corresponding cluster central coordinate
        
"""
def findCluster(cluster,kd_tree,(x,y)):
    ind = int(kd_tree.query([(x,y)],return_distance = False))
    coor = cluster[ind]
    return coor    

"""
Transform the coordinates from form Lambert2 to form WGS84 
    
Agrs:
    (x,y) (tuple): coordinates in form of Lambert2
Returns:
    coor (tuple): coordinates in form of WGS84 -- (latitude,longitude)
        
"""
def transformCoord((x,y)):
    WGS84 = pyproj.Proj(init='epsg:4326')
    Lambert2 = pyproj.Proj(init='epsg:27572')
    lons,lats = pyproj.transform(Lambert2,WGS84,x,y)
    return (lats,lons)

"""
Transform the coordinates from form WGS84 to form Lambert2 
    
Agrs:
    (x,y) (tuple): coordinates in form of WGS84 -- (latitude,longitude)
Returns:
    coor (tuple): coordinates in form of Lambert2
        
"""
def transformLambert((lats,lons)):
    WGS84 = pyproj.Proj(init='epsg:4326')
    Lambert2 = pyproj.Proj(init='epsg:27572')
    x,y = pyproj.transform(WGS84,Lambert2,lons,lats)
    return (x,y)

if __name__ == '__main__':    
    if len(sys.argv) == 3:
        print 'Invalid number of arguments passed.'
        print 'Correct usage: python segmentation.py input_file output_file'
    else:
        seg_file_name = '/appli/bgd/exp_b02/data/bus/72/lig.16_72.csv'
        ref_seg = getSegment(seg_file_name)
        line72 = Line(ref_seg)
        file_name = '/appli/bgd/exp_b02/data/clean_data_gps_20160301_72.csv'
        daily_data = pd.read_csv(file_name)
        daily_data = line72.density_stat(daily_data)
        his = line72.feature_his(0.05)
        features = line72.feature_speed(his,daily_data)
        print features