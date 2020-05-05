# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:36:12 2020

@author: Parssoft
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path

def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

def plot(array,lb,color,title):
    labels=lb
    idx=np.arange(len(labels))   
    plt.figure()
    rect1=plt.bar(idx,array,color = color, width = 0.40)    
    plt.xticks(idx,labels,fontsize='large')
    plt.ylabel('frames')
    #plt.xlabel('sensors')
    
    plt.title(title)
    autolabel(rect1, "center")
    
    
def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    
    for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')
                  
    plt.show()
if __name__ == "__main__":
    df=pd.DataFrame()
    counter=[]
    counter.append(count_files(r'E:\ICT\thesis\sherbrook\\sherbrooke_frames'))
    counter.append(count_files(r'E:\ICT\thesis\rouen\\rouen_frames'))
    counter.append(count_files(r'E:\ICT\thesis\stmark\\stmarc_frames'))
    counter.append(count_files(r'E:\ICT\thesis\rene\\rene'))
    
    names=['Sherbrooke','Rouen','St-Marc','René-Lévesque']
    plot(counter,names,'b','distribution of frames in urban tracker''s dataset')
    chosen=[1001,600,1000,1000]
    plot(chosen,names,'g','distribution of selected frames')
    
    
    