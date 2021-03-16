import requests;
import os
import numpy as np;
import pickle

url='http://127.0.0.1:5000/convert_to_embeddings'
directory='/Users/jatinkaushik/python_projects/final_year_project_ml_work/Face Dataset/train/'
for sub_dir in os.listdir(directory):
	if(sub_dir.find('.DS_Store')!=-1):
		continue;
	images=[];
	for image_name in os.listdir(directory+sub_dir+'/'):
		if(image_name.find('.DS_Store')!=-1):
			continue;
		image_path=directory+sub_dir+'/'+image_name
		images.append(image_path)
	res=requests.post(url,json={'images':images,'roll_no':sub_dir,'class_code':'001'})
	print(res.json());


url='http://127.0.0.1:5000/get_attendance'
students=['Jatin Laklan','Kushal Kumar Mehra','Abhishek rollno-2','ANKUSH','Sumit Solanki'];
class_code='001'
images=['/Users/jatinkaushik/image2.jpg']
res=requests.post(url,json={'images':images,'students':students,'class_code':'001'})
print(res.json());