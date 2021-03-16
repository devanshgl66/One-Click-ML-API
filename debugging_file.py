import requests;
import os
import numpy as np;
import pickle

url='https://ml-one-click-api.herokuapp.com/'
# directory='/Users/jatinkaushik/python_projects/final_year_project_ml_work/Face Dataset/train/'
# for sub_dir in os.listdir(directory):
# 	if(sub_dir.find('.DS_Store')!=-1):
# 		continue;
# 	images=[];
# 	for image_name in os.listdir(directory+sub_dir+'/'):
# 		if(image_name.find('.DS_Store')!=-1):
# 			continue;
# 		image_path=directory+sub_dir+'/'+image_name
# 		images.append(image_path)
images=['https://media.gettyimages.com/photos/hrithik-roshan-attends-the-iifa-2015-press-conference-held-at-grand-picture-id474961324?s=2048x2048']
res=requests.post(url,json={'images':images,'roll_no':17001001024,'class_code':'001'})
print(res);


# url='http://127.0.0.1:5000/get_attendance'
# students=['Jatin Laklan','Kushal Kumar Mehra','Abhishek rollno-2','ANKUSH','Sumit Solanki'];
# class_code='001'
# images=['/Users/jatinkaushik/image2.jpg']
# res=requests.post(url,json={'images':images,'students':students,'class_code':'001'})
# print(res.json());