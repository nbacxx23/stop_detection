#!/usr/bin/env python2

import mpl_toolkits.basemap.pyproj as pyproj
import matplotlib.pyplot as plt
import folium
import operator
from line_model import transformCoord

def generateMap(point):
    trace_points = folium.Map(location=[point[0],point[1]],zoom_start=15,control_scale = True)
    return trace_points

def addCircleMarker(points,map_origin,color,size):
    for point in points:
        x,y = transformCoord(point)
        folium.CircleMarker([x,y],radius=size,fill_color=color).add_to(map_origin)
    return map_origin

def addMarker(points,map_origin):
    for point in points:
        folium.Marker([point[0],point[1]],popup = 'point arret').add_to(map_origin)
    return map_origin

def addColorMapDensity(data,map_origin,colorbar,column):
    for point,value in data.iteritems():
        folium.CircleMarker([transformCoord((point[0],point[1]))[0],transformCoord((point[0],point[1]))[1]],
                            radius=5,fill_color =colorbar(value[column])).add_to(map_origin)
    return map_origin

def colorMapDensity(line,features,linear_color,column):
    locate_point = sorted(line.mini_cluster_stat.items(),key=operator.itemgetter(1),reverse=True)[0][0]
    map_origin = generateMap(transformCoord(locate_point))
    map_origin.add_child(linear_color)
    trace = addColorMapDensity(features,map_origin,linear_color,column)
    return trace
    
def drawCluster(features,map_origin,range_scale):
    poly_list = []
    for cluster in features:
        poly_list.append((transformCoord((cluster[0]-range_scale,cluster[1]-range_scale)),transformCoord((cluster[0]-range_scale,cluster[1]+range_scale))))
        poly_list.append((transformCoord((cluster[0]-range_scale,cluster[1]-range_scale)),transformCoord((cluster[0]+range_scale,cluster[1]-range_scale))))
        poly_list.append((transformCoord((cluster[0]-range_scale,cluster[1]+range_scale)),transformCoord((cluster[0]+range_scale,cluster[1]+range_scale))))
        poly_list.append((transformCoord((cluster[0]+range_scale,cluster[1]+range_scale)),transformCoord((cluster[0]+range_scale,cluster[1]-range_scale))))
    folium.PolyLine(poly_list, color="green", weight=2.5, opacity=1).add_to(map_origin)
    return map_origin