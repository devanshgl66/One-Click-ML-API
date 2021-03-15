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
		if(store_embeddings(file_name,roll_no,embeddings)==None):
			return jsonify({'error':None})
		else:
			return jsonify({'error':True})
	except Exception as e:
		return jsonify({'error':e});





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



def extract_face(filename, required_size=(160, 160)):
     image = Image.open(filename)
     image = image.convert('RGB')
     pixels = asarray(image)
     results = detector.detect_faces(pixels)
     if len(results)==0:
        return (False,None);
     x1, y1, width, height = results[0]['box']
     x1, y1 = abs(x1), abs(y1)
     x2, y2 = x1 + width, y1 + height
     face = pixels[y1:y2, x1:x2]
     image = Image.fromarray(face)
     image = image.resize(required_size)
     face_array = asarray(image)
     return (True,face_array);



def get_embedding(face_pixels):
     face_pixels = face_pixels.astype('float32')
    #  mean, std = face_pixels.mean(), face_pixels.std()
    #  face_pixels = (face_pixels - mean) / std
     face_pixels=face_pixels.reshape(1,160,160,3);
     yhat = model.embeddings(face_pixels)
     return yhat[0]


def give_embeddings(image_links):
    embeddings_list=[];
    for image in image_links:
        flag,face=extract_face(image);
        if(flag):
            embeddings_list.append(get_embedding(face));
    return embeddings_list;




if __name__ == '__main__':
    app.run(debug=True);