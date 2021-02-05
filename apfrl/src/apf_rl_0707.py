#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 00:46:11 2019

@author: kemalbektas
"""
import os
import rospy
import time
import ellipses as el
from utils import add_middle
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from transformations import euler_from_quaternion
from scipy.spatial import ConvexHull
import torch
from model3 import PyTorchMlp
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib.patches import Ellipse
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')                         
class APFRL():
    def __init__(self):
        self.sub_odom = rospy.Subscriber('/diff_vel_controller/odom', Odometry, self.getOdometry)
        self.pub_cmd_vel = rospy.Publisher('/segway/teleop/cmd_vel', Twist, queue_size=10)
        self.vMax = 0.6
        self.wMax = 1*np.pi/4
        self.v_old = 0
        self.w_old = 0
        self.yaw_old= 0
        self.yaw_diff=0
        self.scan_inc = np.pi/512
        self.scan_size = 1024
        self.laser_max = 10
        self.threshold = 0.2
        self.yaw = 0
        self.r = 0.2
        self.r_robot = 0.4
        self.obs_x = []
        self.obs_y = []
        self.obs_a = []
        self.obs_b = []
        self.obs_phi = []
        self.state = []
        self.goals_x = [0]
        self.goals_y = [0]
        self.goal_x = self.goals_x[0]
        self.goal_y = self.goals_y[0]
        self.goal_index = 0
        self.old_scan = []
        self.model = torch.load(os.path.join(os.path.dirname(__file__), '../models/2306_2_586000'))
        self.rate = rospy.Rate(20)
        self.time = 0
        #self.fig, self.axi = plt.subplots(figsize=(10,10))
        #self.axi.set_aspect("equal")
        #plt.ion()
        #plt.show()
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.yaw = euler_from_quaternion(orientation_list)

        if self.yaw<0:
            self.yaw += 2*np.pi
        self.yaw_diff = self.yaw - self.yaw_old
        if self.yaw_diff>np.pi:
            self.yaw_diff = 2*np.pi - self.yaw_diff
        elif self.yaw_diff<-np.pi:
            self.yaw_diff = -2*np.pi - self.yaw_diff

        self.yaw_old = self.yaw

    def findComponents(self,scan):
        start_points = [0]    
        end_points = [0]   
        count = 1
        #for i in range(512):
        #    self.scan.append(self.scan[i]) # WHY ??
        #del self.scan[:int(self.scan_size/2)]
        for i in range(self.scan_size-1):
            diff = np.abs(scan[i+1]-scan[i])
            avg = (scan[i+1]+scan[i])/2
            
            if(diff > self.threshold*0.75):
                count += 1
                start_points.append(i+1)
                end_points.append(i+1)
            else:
                end_points[count-1] = i
                
        
            
        if(end_points[-1]>1020):
            start_points[0] = start_points[-1]-1024
            del(end_points[-1])
            del(start_points[-1])
        
        count_2 = 0
        for i in range(len(start_points)):
            if (self.laser_max - np.mean(scan[start_points[i-count_2]:end_points[i-count_2]])) < 0.5 or end_points[i-count_2]-start_points[i-count_2]<4:
                del start_points[i-count_2]
                del end_points[i-count_2]
                count_2 += 1
                
        return start_points,end_points
    
    
    def findEllipses(self,scan,start_points,end_points):
        self.obs_x = []
        self.obs_y = []
        self.obs_a = []
        self.obs_b = []
        self.obs_phi = []
        x = scan*np.cos(np.pi/(self.scan_size / 2)*(np.arange(self.scan_size)-self.scan_size/2))
        y = scan*np.sin(np.pi/(self.scan_size / 2)*(np.arange(self.scan_size)-self.scan_size / 2))
        ax = np.cos(self.yaw)*x - np.sin(self.yaw)*y
        ay = np.sin(self.yaw)*x + np.cos(self.yaw)*y
        #plt.scatter(ax,ay)
        points = np.dstack([x,y])[0]
        points = points[np.linalg.norm(points,axis=1)<10]
        hull = ConvexHull(points)
        vx = []
        vy = []
        hull.simplices = hull.simplices[hull.simplices[:,0].argsort()]
        for simplex in hull.simplices:
            #plt.plot(points[simplex, 0], points[simplex, 1],'k-')
            points[simplex, 0][0]
            points[simplex, 0][0]
            vx.append(points[simplex, 0][0])
            vx.append(points[simplex, 0][1])
            vy.append(points[simplex,1][0])
            vy.append(points[simplex,1][1])
                
        nearest = np.argmin(np.linalg.norm([vx,vy],axis=0))
        vp = add_middle([vx,vy],nearest,2)
        
        lsqe = el.LSqEllipse()
        lsqe.fit(vp)
        #plt.plot(vp[0],vp[1])
        [o_x,o_y], o_a, o_b, o_phi = lsqe.parameters()
        self.obs_x.append(o_x)
        self.obs_y.append(o_y)
        self.obs_a.append(o_a)
        self.obs_b.append(o_b)
        self.obs_phi.append(o_phi+self.yaw)
        
        for i in range(len(start_points)):
            if start_points[i]<0 and end_points[i]<0:
                start_points[i] += 1024
                end_points[i] += 1024
            if start_points[i]<0:
                ri = np.array(scan[start_points[i]:]+scan[:end_points[i]])
                xi = np.array(x[start_points[i]:].tolist()+x[:end_points[i]].tolist())
                yi = np.array(y[start_points[i]:].tolist()+y[:end_points[i]].tolist())
                #axi = np.array(ax[start_points[i]:].tolist()+ax[:end_points[i]].tolist())
                #ayi = np.array(ay[start_points[i]:].tolist()+ay[:end_points[i]].tolist())
            else:
                ri = scan[start_points[i]:end_points[i]] 
                xi = x[start_points[i]:end_points[i]] 
                yi = y[start_points[i]:end_points[i]]
                #axi = ax[start_points[i]:end_points[i]] 
                #ayi = ay[start_points[i]:end_points[i]]
            #plt.scatter(xi,yi)
            if(np.any(np.in1d(np.arange(start_points[i],end_points[i]), hull.vertices))) or abs(start_points[i] - end_points[i])<3 \
            or min(ri) > 1.5: #Consider a change
                continue
            else:
                
                try:
                    lsqe = el.LSqEllipse()
                    lsqe.fit([xi,yi])
            
                    [o_x,o_y], o_a, o_b, o_phi = lsqe.parameters()
                    if np.iscomplex(o_x) or np.iscomplex(o_y) or np.iscomplex(o_a) or np.iscomplex(o_b) or np.iscomplex(o_phi) \
                    or o_a<0.05 or o_b<0.05 or o_a > 1.5 or o_b >1.5 or o_a*o_b>1:
                        a = zaa 
                except:
                    o_x = (xi[0]+xi[-1])/2
                    o_y = (yi[0]+yi[-1])/2
                    o_a = 0.55*np.sqrt((xi[0]-xi[-1])**2 + (yi[0]-yi[-1])**2)
                    o_b = min(0.25,o_a)
                    o_phi = np.arctan2(yi[-1]-yi[0],xi[-1]-xi[0])                   
        
                self.obs_x.append(o_x)
                self.obs_y.append(o_y)
                self.obs_a.append(o_a)
                self.obs_b.append(o_b)
                self.obs_phi.append(o_phi+self.yaw)
            
#        for i in range(len(self.obs_x)):
#            if i==0:
#                oax = np.cos(self.yaw)*self.obs_x[i] - np.sin(self.yaw)*self.obs_y[i]
#                oay = np.sin(self.yaw)*self.obs_x[i] + np.cos(self.yaw)*self.obs_y[i]
#                c = Ellipse(xy=(oax,oay), width=2*self.obs_a[i], height=2*self.obs_b[i], angle=np.rad2deg(self.obs_phi[i]),edgecolor='r', fc='None', lw=2, label='Fit', zorder = 2)
#               self.axi.add_patch(c)
#        c = plt.Circle((0,0), 0.2,color='r')
#        self.axi.add_artist(c)
    
#        plt.xlim([-20,20])
#        plt.ylim([-20,20])
#        plt.pause(0.5)
#        plt.cla()
                
        return self.obs_x,self.obs_y,self.obs_a,self.obs_b,self.obs_phi

    def reinitialize(self):
        self.goal_x = 0
        self.goal_y = 0


    def getState(self, scan):
        scan_range = []
        min_range = 0.05 + self.r_robot
        done = False
        col = False
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(self.laser_max)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        #scan_range = np.roll(scan_range,512).tolist()
         
        if min_range > np.min(scan_range) > 0:
            col = True
            done = True
            print("collision!!")
            
        start,end = self.findComponents(scan_range)
        self.findEllipses(scan_range,start,end)
        
        current_distance = np.linalg.norm([self.goal_x - self.position.x, self.goal_y - self.position.y])
        dx = self.goal_x - self.position.x
        dy = self.goal_y - self.position.y
        gx = dx*np.cos(self.yaw) + dy*np.sin(self.yaw)
        gy = -dx*np.sin(self.yaw) + dy*np.cos(self.yaw)
        
        scan_t = (np.array(scan_range[::16]) - self.r_robot).tolist()
        
        if self.old_scan == []:
            self.old_scan = scan_t + [gx,gy]
        if current_distance < 0.3:
            print("Goal!!")
            self.goal_index += 1 
            if not self.goal_index >=len(self.goals_x):
                self.goal_x = self.goals_x[self.goal_index]
                self.goal_y = self.goals_y[self.goal_index]
            else:
                print("All done!")
                done = True
        shift = int(self.yaw_diff*32/np.pi)
        self.old_scan[:64] = np.roll(self.old_scan[:64], shift).tolist()
        self.state = ((np.array(self.old_scan + scan_t + [gx, gy]))/(self.laser_max)).tolist()
        self.old_scan = scan_t + [gx, gy]
        return self.state,col,done
    def step(self):
        if self.time == 0:
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                except Exception as e:
                    print(e)
                    print("No scan")
                    if rospy.is_shutdown():
                        break
            while self.goal_x == 0 and self.goal_y == 0:
                try:
                    goal = rospy.wait_for_message('goal', PoseStamped, timeout=5)
                    self.goal_x = goal.pose.position.x
                    self.goal_y = goal.pose.position.y
                except:
                    print("No goal!")
                    pass
                    if rospy.is_shutdown():
                        break
        state,col,done = self.getState(data)
        with torch.no_grad():
            action = self.model(torch.from_numpy(np.array(state)).float()).numpy()   
        print("action: " + str(action[0]) + ","+str(action[1]) + ","+str(action[2]))
        self.target=[]
        wp_done = False
        for i in range(15):
            linear_vel,ang_vel,wpDone = self.apf(action)     
            if min(self.state[123:131]+self.state[66:74])<0.02:
                linear_vel = min(0.1, linear_vel)
            if min(self.state[74:90])<0.02:
                ang_vel = min(0.1, ang_vel)
            if min(self.state[91:107])<0.02:
                ang_vel = max(-0.1, ang_vel)
            vel_cmd = Twist()
            vel_cmd.linear.x = linear_vel 
            vel_cmd.angular.z = ang_vel
            self.rate.sleep()
            self.pub_cmd_vel.publish(vel_cmd)
            time.sleep(0.05)
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                except Exception as e:
                    print(e)
                    print("No scan")
                    if rospy.is_shutdown():
                        break
            state, col,done = self.getState(data)
            #self.rate.sleep()
            if done:
                self.pub_cmd_vel.publish(Twist())
                break
            elif wp_done:
                break
        return done
    def apf(self,action):
        ori = self.yaw
        current_distance = np.linalg.norm([self.goal_x - self.position.x,
                                           self.goal_y - self.position.y])
        if current_distance<1:
            dgx = self.goal_x - self.position.x
            dgy = self.goal_y - self.position.y
            action[2] = 0.5
        elif not self.target==[]:
            dgx = self.target[0] - self.position.x
            dgy = self.target[1] - self.position.y
        else:
            dgx = (np.cos(ori)*(action[0]+1)/2 - np.sin(ori)*action[1])
            dgy = (np.sin(ori)*(action[0]+1)/2 + np.cos(ori)*action[1])
            self.target = [dgx+self.position.x, dgy+self.position.y]

        dbx = []
        dby = []
        denomx = 0
        denomy = 0
        Bs = []
        self.obs_x = np.real(self.obs_x)
        self.obs_y = np.real(self.obs_y)
        self.obs_a = np.real(self.obs_a)
        self.obs_b = np.real(self.obs_b)
        self.obs_phi = np.real(self.obs_phi)
        dx = np.array(-(np.cos(ori)*self.obs_x - np.sin(ori)*self.obs_y))
        dy = np.array(-(np.sin(ori)*self.obs_x + np.cos(ori)*self.obs_y))
        oth = np.arctan2(dy,dx) - self.obs_phi
        obs_r = 1.1*(self.obs_a*self.obs_b/np.sqrt(self.obs_a**2*np.sin(oth)**2+self.obs_b**2*np.cos(oth)**2))
        obs_r[0] /= 1.1
        dth = 1.1*(self.obs_a*self.obs_b*np.sin(oth)*np.cos(oth)*(self.obs_a**2-self.obs_b**2)/(self.obs_a**2*np.sin(oth)**2+self.obs_b**2*np.cos(oth)**2)**(3/2))
        dth[0] /= 1.1
        dtx = -dy/(dx**2+dy**2)
        
        dty = dx/(dx**2+dy**2)
        for i in range(len(self.obs_x)):
            if(i == 0):
                b = -dx[i]**2-dy[i]**2 + (obs_r[i]-self.r_robot)**2
                Bs.append(b)
                dbx.append(-2*dx[i]+2*(obs_r[i]-self.r_robot)*dth[i]*dtx[i])
                dby.append(-2*dy[i]+2*(obs_r[i]-self.r_robot)*dth[i]*dty[i])
                denomx += dbx[i]/b
                denomy += dby[i]/b
            else:
                b = dx[i]**2+dy[i]**2 - (self.r_robot+obs_r[i])**2
                #if np.sqrt(b)<1.3:
                Bs.append(b)
                dbx.append(2*dx[i]-2*(obs_r[i]+self.r_robot)*dth[i]*dtx[i])
                dby.append(2*dy[i]-2*(obs_r[i]+self.r_robot)*dth[i]*dty[i])
                denomx += dbx[-1]/b
                denomy += dby[-1]/b
        gamma = dgx**2+dgy**2
        
        dgammax = 2*dgx
        dgammay = 2*dgy
        k = (action[2]+1.5)*2
        B = np.prod(Bs)
        #k = round(np.log(B) / np.log(gamma))
        Fth = np.arctan2(-(k*dgammay*B - gamma*B*denomy),-(k*dgammax*B - gamma*B*denomx))
        th = Fth - self.yaw - np.pi
        v = max((action[0]+1)*np.cos(th)*self.vMax/2,np.cos(th)*self.vMax/2)
        w=max(abs(action[1])*np.sin(th)*self.wMax,np.sin(th)*self.wMax/2)

        if np.sqrt(gamma)<0.1:
            done = True
        else:
            done = False
        return v,w,done

        
if __name__ == '__main__':
    rospy.init_node('apfrl')
    done = False
    apf = APFRL()
    while not (done and rospy.is_shutdown()):
        done = apf.step()
        if done:
            apf.reinitialize()

        

        
