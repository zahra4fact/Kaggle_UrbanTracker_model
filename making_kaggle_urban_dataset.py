# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:11:57 2020

@author: Parssoft
"""

import os , shutil
import random 
from PIL import Image


p=r"E:\ICT\thesis\kaggle\new"
proen=r"E:\ICT\thesis\urban tracker\frames\rouen\rouen_frames"
pshare=r"E:\ICT\thesis\urban tracker\sherbrook\sherbrooke_frames"
pst=r"E:\ICT\thesis\urban tracker\stmark\stmarc_frames"

purban_frame=r"E:\ICT\thesis\urban tracker\frames"
RootDir1 = r"E:\ICT\thesis\kaggle\aaurainsnow"
TargetFolder = r'E:\ICT\thesis\kaggle\Data'
data=r"E:\ICT\thesis\data\public data\train\good_weather"

def moving(psource,pdestination):
    for root, dirs, files in os.walk((os.path.normpath(psource)), topdown=False):
        for name in files:
            dst =  "Class_2_" + str(count) + ".png"
            if name.endswith('.png'):
                print("found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, pdestination) 
            
           
        
                
def select_randomly(path,number):
    files=os.listdir(path)
    img =random.sample(list(enumerate(files)),number)
    print(img)
#    index=[]
    index=[x[1] for x in img]
#    for i in range(len(index)):
#        del files[i] 
    for f in index:
        print(path+'\\'+ f)
        os.remove(path+'\\'+ f)
    print("success")
        
select_randomly(data,50)
#select_randomly(proen,88)
#select_randomly(pshare,540)
#select_randomly(pst,540)
moving(RootDir1,data)
#count=0
#for root, dirs, files in os.walk((os.path.normpath(purban_frame)), topdown=False):
#    print(files)
#    for name in files:
#        dst =  "Class_2_" +str(count) + ".jpg"
#        if name.endswith('.jpg'):
#            SourceFolder = os.path.join(root,name)
#            shutil.copy2(SourceFolder, data) 
#            count+=1
#os.getcwd()
#c=1
#collection = r"E:\ICT\thesis\kaggle\Data"
#dest=r"E:\ICT\thesis\kaggle\new"
#for i, filename in enumerate(os.listdir(collection)):
#    os.rename(collection +'\\' +filename, dest +'\\'+'bad_weather' +str(c) + ".jpg")
#    c+=1
#g=r"E:\ICT\thesis\data\public data\good weather"
#select_randomly(g,100)
    
#directory = r'E:\ICT\thesis\data\public data\good weather'
#c=1
#for filename in os.listdir(directory):
#    if filename.endswith(".jpg"):
#        im = Image.open(directory+'\\'+filename)
#        name='img'+str(c)+'.png'
#        rgb_im = im.convert('RGB')
#        rgb_im.save(name)
#        c+=1
#        continue
#    else:
#        continue