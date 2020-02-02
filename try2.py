from keras.models import load_model
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
from keras_vggface.utils import preprocess_input
import numpy as np
from cropper import extract_face
import warnings

warnings.filterwarnings('ignore')

model = load_model('face_model.h5')

#test_img = load_img('test.jpg', target_size=(224, 224))
#img_test = img_to_array(test_img)
img_test = extract_face('test.jpg')
img_test = img_test.astype('float32')
img_test = np.expand_dims(img_test, axis=0)
img_test = preprocess_input(img_test)
predictions = model.predict(img_test)
predicted_class=np.argmax(predictions,axis=1)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                        './dataset',
                        target_size=(224,224),
                        batch_size=5,
                        class_mode='sparse',
                        color_mode='rgb')


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class]
print('Found the following actor on this photo:')
print(predictions)