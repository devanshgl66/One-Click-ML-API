from flask import Flask,jsonify,request
from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import load
from numpy import asarray
from keras_facenet import FaceNet
from keras.models import load_model
import cv2;
import numpy as np;
import pickle;
import requests
detector = MTCNN()
model = FaceNet() 



app=Flask(__name__)


@app.route("/")
def home():
    return jsonify({'ndjc':'ascc'});


@app.route("/convert_to_embeddings",methods=['POST'])
def convert_to_embeddings():
	try:
		data=request.get_json(force=True)
		roll_no=data['roll_no']
		images=data['images']
		class_code=data['class_code']
		embeddings=give_embeddings(images)
		file_name=class_code+'.p';
		error=store_embeddings(file_name,roll_no,embeddings)
		if(error==None):
			return jsonify({'error':None})
		else:
			return jsonify({'error':error})
	except Exception as e:
		return jsonify({'error':e});


@app.route("/get_attendance",methods=['POST'])
def get_attendance():
	try:
		data=request.get_json(force=True)
		students=data['students']
		class_code=data['class_code']
		images=data['images'];
		embeddings=give_embeddings(images)
		file_name=class_code+'.p'
		students=find_present_students(embeddings,file_name,students)
		return jsonify({
			'students':students,
			'error':None,
			'success':True
			})
	except Exception as e:
		return jsonify({
			'success':False,
			'error':e,
			'students':[]
			})




def find_present_students(embeddings,file_name,students):
	file=open(file_name,'rb')
	dictionary=pickle.load(file);
	present_list=[];
	for embedding in embeddings:
		min_dist=1000000
		roll_no=None
		for student in students:
			temp_dist=find_dist(embedding,dictionary[student]);
			if(temp_dist<min_dist):
				min_dist=temp_dist
				roll_no=student
		present_list.append(roll_no)
	return present_list



def find_dist(embedding,student_embeddings):
	min_dist=1000000
	for single_embedding in student_embeddings:
		temp_dist=np.linalg.norm(single_embedding-embedding)
		if(temp_dist<min_dist):
			min_dist=temp_dist
	return min_dist



def store_embeddings(file_name,roll_no,embeddings):
	try:
	    new_dict = {roll_no:embeddings}
	    try:
	        file=open(file_name, 'rb')
	        old_data=pickle.load(file);
	    except:
	        dictionary={};
	        file=open(file_name, 'wb')
	        pickle.dump(dictionary,file) 
	        file.close()
	    finally:
	        file=open(file_name, 'rb')
	        old_data=pickle.load(file);
	        new_dict.update(old_data)
	        file=open(file_name, 'wb')
	        pickle.dump(new_dict,file) 
	except Exception as e:
		return e;



def extract_faces(url, required_size=(160, 160)):
     image = Image.open(requests.get(url, stream=True).raw)
     image = image.convert('RGB')
     pixels = asarray(image)
     results = detector.detect_faces(pixels)
     faces_array=[];
     if len(results)==0:
        return (False,None);
     for info in results:
     	x1, y1, width, height = info['box']
     	x1 = abs(x1)
     	y1 = abs(y1)
     	x2, y2 = x1 + width, y1 + height
     	face = pixels[y1:y2, x1:x2]
     	image = Image.fromarray(face)
     	image = image.resize(required_size)
     	faces_array.append(asarray(image))
     return (True,faces_array);



def get_embedding(face_pixels):
     face_pixels = face_pixels.astype('float32')
     face_pixels=face_pixels.reshape(1,160,160,3);
     yhat = model.embeddings(face_pixels)
     return yhat[0]



def give_embeddings(image_links):
    embeddings_list=[];
    for image in image_links:
        flag,faces_array=extract_faces(image);
        if(flag):
        	for face in faces_array:
        		embeddings_list.append(get_embedding(face));
    return embeddings_list;




if __name__ == '__main__':
    app.run(debug=False);