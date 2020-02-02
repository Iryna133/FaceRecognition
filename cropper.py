from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os
import fnmatch

def extract_face(filename, required_size=(224, 224)):
	pixels = pyplot.imread(filename)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

filelist = fnmatch.filter(os.listdir('./'), '*.jpg')
if not os.path.exists('./cropped'):
    os.makedirs('./cropped')
for pic in filelist:
    print('Croping: ' + pic)
    final_pic = Image.fromarray(extract_face(pic))
    final_pic.save('./cropped/' + pic)
