#!/usr/bin/env python
"""
Region Prediction Algorithm using Agg backend for matplotlib

This script reads fingerprint data, processes test points, performs region prediction 
based on highest RSSI for a given list of BSSIDs, calculates errors and prediction times, 
and finally displays/saves the result image.
"""

# Force matplotlib to use the non-interactive Agg backend (instead of Tkinter)
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import os

from common_functions import read_fingerprint_data, cdf, calculate_results, write_results, distance, sortonerror, sort_dict_by_rssi, show_save_image

# Global parameters
img = cv2.imread("images/HV I-J Plan 1.png")
k = 5
database = "fpData-Full.txt"
csv_path = "CSV/Test All.csv"
test = "b_All_Tests/"

def regionPredictionAlgorithm(img, k, database, csv_path, test):
    start_total_time = time.time()  # start recording total run time

    # Given list of BSSIDs to consider
    bssid_list = ['70:b3:17:8d:e9:60', '70:b3:17:8e:1c:00', '78:bc:1a:37:7e:00',
                  '48:8b:0a:ca:a8:00', '48:8b:0a:cb:67:e0', '48:8b:0a:cb:69:20']

    # Read fingerprint (reference) data from fpData-Full.txt
    fingerprint_data = read_fingerprint_data(database)
    # Determine, for each reference point, which BSSID in our list has the highest RSSI
    regions = find_highest_rssi_bssid(fingerprint_data, bssid_list)
    # Sort region data: region_select will hold region identifiers and r_data holds corresponding fingerprint data
    region_select, r_data = sort_region_data(regions, bssid_list, fingerprint_data)

    # (Optional) Draw circles on the image for each reference point
    # draw_circles(img, regions)

    ErrorList = []
    PredictionTime = []
    file_name = os.path.basename(__file__)

    csv_filepath = csv_path
    PredictionTime, ErrorList = open_CSV(csv_filepath, img, k, ErrorList, PredictionTime, region_select, r_data, bssid_list, test)

    end_total_time = time.time()  # end recording total run time
    total_time = end_total_time - start_total_time

    # (Optional) Display and save the image using a function that uses Agg (not Tkinter)
    # show_save_image(img)

    return PredictionTime, ErrorList, total_time, file_name, k

def find_highest_rssi_bssid(fingerprint_data, bssid_list):
    highest_rssi_bssid_info = {}
    for key, data in fingerprint_data.items():
        highest_rssi = float('-inf')  # Initialize with negative infinity
        highest_rssi_bssid = None
        for bssid in bssid_list:
            truncated_bssid = bssid[:14]  # Consider only the first 14 characters
            if any(truncated_bssid in full_bssid for full_bssid in data):
                # Check if any BSSID in data starts with the truncated BSSID
                max_rssi_for_truncated_bssid = max(data[full_bssid] for full_bssid in data if truncated_bssid in full_bssid)
                if max_rssi_for_truncated_bssid > highest_rssi:
                    highest_rssi = max_rssi_for_truncated_bssid
                    highest_rssi_bssid = truncated_bssid
        highest_rssi_bssid_info[key] = highest_rssi_bssid
    return highest_rssi_bssid_info

def get_color(truncated_bssid):
    """
    Assigns a color based on the given BSSID.
    """
    color_map = {
        '70:b3:17:8d:e9': (255, 0, 0),    # Red
        '70:b3:17:8e:1c': (0, 255, 0),    # Green
        '78:bc:1a:37:7e': (0, 0, 255),     # Blue
        '48:8b:0a:ca:a8': (255, 255, 0),   # Yellow
        '48:8b:0a:cb:67': (255, 0, 255),   # Magenta
        '48:8b:0a:cb:69': (0, 255, 255)    # Cyan
    }
    for prefix, color in color_map.items():
        if truncated_bssid.startswith(prefix):
            return color
    return (128, 128, 128)  # Default color if prefix not found

def filter_region(regions, bssid_value):
    filtered_region = {}
    for key, bssid in regions.items():
        if bssid == bssid_value:
            filtered_region[key] = bssid
    return filtered_region

def extract_region_data(fp_data, filter_region):
    extracted_data = {}
    for key, data in fp_data.items():
        if key in filter_region:
            extracted_data[key] = data
    return extracted_data

def sort_region_data(regions, bssid_list, fingerprint_data):
    region_select = []
    r_data = []  # Contains fingerprint data for each region
    for i in range(len(bssid_list)):
        r1 = filter_region(regions, bssid_list[i][:14])
        r1_data = extract_region_data(fingerprint_data, r1)
        r_data.append(r1_data)
        r1 = [bssid_list[i][:14], r1]
        region_select.append(r1)
    return region_select, r_data

def GetCandidatePos(online, fpdb, temp_k):
    candidates = []
    for key in fpdb.keys():
        candidate = fpdb[key]
        errRSSI = []
        k = temp_k  # k is the number of neighboring WiFi access points to consider
        for bssid in candidate.keys():
            if bssid in online.keys():
                if k == 0:
                    break
                k -= 1
                errRSSI.append(abs(candidate[bssid] - online[bssid]))
        if errRSSI:
            average = sum(errRSSI) / len(errRSSI)
            candidates.append([key, average])
    candidates.sort(key=sortonerror)
    return candidates[0]

def draw_circles(img, regions):
    for key, bssid in regions.items():
        loc = key.split('_')  # split the key into x and y coordinates
        posX = int(loc[0])
        posY = int(loc[1])
        color = get_color(bssid)
        cv2.circle(img, (posX, posY), 4, color, 2)

def open_CSV(filepath, img, temp_k, ErrorList, PredictionTime, region_select, r_data, bssid_list, test_map):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for gtPtInfo in reader:
            gtPtX = int(gtPtInfo[1])
            gtPtY = int(gtPtInfo[2])
            gtPtFile = gtPtInfo[3]
            fileName = gtPtFile
            listScans = []
            test_filepath = test_map + fileName + ".txt"
            start_time = time.time()
            open_test(test_filepath, listScans, gtPtX, gtPtY, img, temp_k, ErrorList, region_select, r_data, bssid_list)
            end_time = time.time()
            duration = end_time - start_time
            PredictionTime.append(duration)
    return PredictionTime, ErrorList

def open_test(filepath, listScans, gtPtX, gtPtY, img, temp_k, ErrorList, region_select, r_data, bssid_list):
    with open(filepath, 'r') as fileGT:
        reader = csv.reader(fileGT, delimiter=';')
        scanid = 0
        prvTag = "NONE"
        listBSSIDnRSSI = {}
        for apInfo in reader:
            if len(apInfo) > 4:
                if apInfo[0] == "WIFI":
                    scanid = apInfo[2]
                    bssid = apInfo[4]
                    rssi = int(apInfo[5])
                    listBSSIDnRSSI[bssid] = rssi
                elif prvTag == "WIFI" and apInfo[0] != "WIFI":
                    listScans.append([scanid, listBSSIDnRSSI])
                    listBSSIDnRSSI = {}
                prvTag = apInfo[0]
        listBSSIDnRSSI = listScans[0][1]
        tpMap = {}
        tpPosition = str(gtPtX) + "_" + str(gtPtY)
        tpMap[tpPosition] = listBSSIDnRSSI
        tpRegion = find_highest_rssi_bssid(tpMap, bssid_list)
        selected_region = select_region(gtPtX, gtPtY, tpRegion, region_select, r_data, img)
        if listScans:
            listScans[-1][1] = sort_dict_by_rssi(listScans[-1][1])
        listScans = [[1, listBSSIDnRSSI]]
        for scan in listScans:
            onlineScan = scan[1]
            locXY, errRSSI = GetCandidatePos(onlineScan, selected_region, temp_k)
            loc = locXY.split('_')
            posX = int(loc[0])
            posY = int(loc[1])
            cv2.circle(img, (posX, posY), 4, (0, 0, 0), 2)
            cv2.line(img, (gtPtX, gtPtY), (posX, posY), (0, 255, 0), 2)
            ErrorList.append(abs(distance((posX, posY), (gtPtX, gtPtY)) / 35.7))

# Helper function: selects the region for a test point
def select_region(posX, posY, test_Point_Region, region_select, region_data, img):
    selected_region = {}
    key, value = next(iter(test_Point_Region.items()))
    for i in range(len(region_select)):
        if region_select[i][0] == value:
            selected_region = region_data[i]
            color = get_color(value)
            cv2.circle(img, (posX, posY), 4, color, 2)
            return selected_region

PredictionTime, ErrorList, total_time, file_name, k = regionPredictionAlgorithm(img, k, database, csv_path, test)
standard_deviation_error = np.std(ErrorList)
average_error = sum(ErrorList) / len(ErrorList)
print("Average Error: ", average_error)
print("Standard Deviation Error: ", standard_deviation_error)
print("Total Time: ", total_time)
print("File Name: ", file_name)
print("K Nearest Neighbors: ", k)

calculate_results(total_time, ErrorList, PredictionTime, k, file_name, database)
show_save_image(img)
