from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

print("FACE MODELS:\n")
face_model = VGGFace(model='vgg16', 
                weights='vggface',
                input_shape=(224,224,3)) 

print("face_model SUMMARY:\n")
face_model.summary()

for layer in face_model.layers:
    layer.trainable = False

person_count = 14

last_layer = face_model.get_layer('pool5').output

x = Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc6')(x)
x = Dense(1024, activation='relu', name='fc7')(x)
out = Dense(person_count, activation='softmax', name='fc8')(x)

custom_face = Model(face_model.input, out)

print("custom_face SUMMARY:\n")
custom_face.summary()

print("IMAGES PREPROCESSING:\n")
batch_size = 5
train_path = 'dataset/'
eval_path = 'eval/'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(224,224),
                        batch_size=batch_size,
                        class_mode='sparse',
                        color_mode='rgb')

valid_generator = valid_datagen.flow_from_directory(
    directory=eval_path,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True)

print("TRAINING TIME!\n")
custom_face.compile(loss='sparse_categorical_crossentropy',
                         optimizer=SGD(lr=1e-4, momentum=0.9),
                         metrics=['accuracy'])

history = custom_face.fit_generator(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=49/batch_size,
        validation_steps=valid_generator.n,
        epochs=50)

custom_face.evaluate_generator(generator=valid_generator, steps=10)
        
custom_face.save('face_model.h5')