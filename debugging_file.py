import requests;
import os
import numpy as np;
import pickle

url='https://ml-one-click-api.herokuapp.com/convert_to_embeddings'
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
images=['https://image.shutterstock.com/shutterstock/photos/1714665613/display_1500/stock-photo-thoughtful-businessman-wearing-glasses-touching-chin-pondering-ideas-or-strategy-sitting-at-1714665613.jpg']
res=requests.post(url,json={'images':images,'roll_no':17001001024,'class_code':'001'})
print(res);


# url='http://127.0.0.1:5000/get_attendance'
# students=['Jatin Laklan','Kushal Kumar Mehra','Abhishek rollno-2','ANKUSH','Sumit Solanki'];
# class_code='001'
# images=['/Users/jatinkaushik/image2.jpg']
# res=requests.post(url,json={'images':images,'students':students,'class_code':'001'})
# print(res.json());