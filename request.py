import requests;

url='http://127.0.0.1:5000/convert_to_embeddings'
path='/Users/jatinkaushik/python_projects/final_year_project_ml_work/Face Dataset/train/Jatin Kaushik/'
images=[path+'1st.jpg',path+'2nd.jpg',path+'3rd.jpg'];
res=requests.post(url,json={'images':images,'roll_no':17001001024,'class_code':'QAWXE'});
print(res.json());