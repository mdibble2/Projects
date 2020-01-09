# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 19:06:08 2018

@author: Megan Dibble
"""
#format for installing python libraries on the vm: sudo python -m pip install sklearn

import csv
import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.cluster import DBSCAN
from sklearn import metrics

#Open the file from the command prompt, name the variable csv_file
csv_file = open(sys.argv[1], 'rb')
csvReader = csv.reader(csv_file)

# Get the header, or first line of the file, and find out where the
# lon and lat are located in the data (look in the file, they are the
# 2nd and 3 elements in the header, which is index 1 & 2)
#header = csvReader.next()
#dateandtimestampIndex = header.index("dateandtimestamp")
#lonIndex = header.index("lon")
#latIndex = header.index("lat")
#satsIndex = header.index("sats")
#tempIndex = header.index("temp")

# After install, import more libraries
# These must be installed using the command above.
from geopy.distance import great_circle 
from shapely.geometry import MultiPoint

# define the number of kilometers in one radian, a needed paramter
kms_per_radian = 6371.0088

# load the data set
df = pd.read_csv('csv_file', encoding='utf-8')

# To prepare for the DBSCAN, represent points consistently as (lat, lon) in a matrix data structure
coords = df.as_matrix(columns=['lat', 'lon'])

# define DBSCAN input paramters epsilon and min_samples
# Read about the paramters here: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

# epsilon is your guess on the max distance between stationary points
# the units are in kilometers, so .03 would be .03 kilometers or 30 meters
epsilon = 0.015 / kms_per_radian 

# minimum_samples, this is the number of datapoints required to be considered "stationary"
# Remember, each datapoins is 1 second apart, so 30 or 60 (1 minute) or 300 (5 minutes)?
ms = 30 # this represents 30 seconds in one place to be a stationary_point

# This is the cell that calls the DBSCAN algorithm. This may take a long time, depending on
# the number of points in your datafile. You must wait for the * in the In[*] to complete.
# On a big dataset, this took about 5 minutes on my pretty powerful desktop

start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=ms, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

# Add cluster labels column to original dataframe
# so we know if each point is a "stationary" point (does NOT have a value of -1)
# or if each point is a "travel" point (has a value of -1)
df['cluster_labels']=cluster_labels

# Just get the values where cluster_labels = -1
# These are the "travel points"
travel_points = df.loc[df['cluster_labels'] == -1]

# Save stationary and travel points out to .csv files, in case you want to plot them in something else
filename = os.path.splitext(sys.argv[1])[0]
travel_points.to_csv(filename+'_stationary_points.csv', encoding='utf-8')

csv_file.close()
