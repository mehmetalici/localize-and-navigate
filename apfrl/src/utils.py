#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:54:41 2020

@author: kemalbektas
"""
import numpy as np

def add_middle(points, index=-1, loop=1):
    if index == -1:
        vx = np.array([])
        vy = np.array([])
        x_inc = (np.roll(points[0], 1) - points[0])/(loop+1)
        y_inc = (np.roll(points[1], 1) - points[1])/(loop+1)
        for i in range(loop):
            vx = np.concatenate([vx, points[0]+x_inc*(i+1)])
            vy = np.concatenate([vy, points[1]+y_inc*(i+1)])
        points[0] = np.concatenate([vx, points[0]]).tolist()
        points[1] = np.concatenate([vy, points[1]]).tolist()

    else:
        x1_inc = (points[0][(index+1)%len(points[0])]-points[0][index])/(loop+1)
        x2_inc = (points[0][(index-1)%len(points[0])]-points[0][index])/(loop+1)
        x3_inc = (points[0][(index+2)%len(points[0])]-points[0][index])/(loop+1)
        x4_inc = (points[0][(index-2)%len(points[0])]-points[0][index])/(loop+1)
        y1_inc = (points[1][(index+1)%len(points[1])]-points[1][index])/(loop+1)
        y2_inc = (points[1][(index-1)%len(points[1])]-points[1][index])/(loop+1)
        y3_inc = (points[1][(index+2)%len(points[1])]-points[1][index])/(loop+1)
        y4_inc = (points[1][(index-2)%len(points[1])]-points[1][index])/(loop+1)
        for i in range(loop):
            points[0].append(points[0][index]+x1_inc*(i+1))
            points[0].append(points[0][index]+x2_inc*(i+1))
            points[0].append(points[0][index]+x3_inc*(i+1))
            points[0].append(points[0][index]+x4_inc*(i+1))
            points[1].append(points[1][index]+y1_inc*(i+1))
            points[1].append(points[1][index]+y2_inc*(i+1))
            points[1].append(points[1][index]+y3_inc*(i+1))
            points[1].append(points[1][index]+y4_inc*(i+1))
    return points

def fill_gap(r):
    far = False
    count = 0
    for i in range(len(r)):
        if r[i] > 0.08:
            far = True
        else:
            far = False
        if far:
            count += 1
        elif count>0 and count<=8:
            r[i-count:i] = (r[i]+r[i-count-1])/2
            count = 0
        else:
            count = 0
    return r
