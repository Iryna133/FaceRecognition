from keras.optimizers import SGD
import model

custom_face.compile(loss='sparse_categorical_crossentropy',
                         optimizer=SGD(lr=1e-4, momentum=0.9),
                         metrics=['accuracy'])

history = custom_face.fit_generator(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=49/batch_size,
        validation_steps=valid_generator.n,
        epochs=50)

custom_face.evaluate_generator(generator=valid_generator)
        
custom_face.save('vgg_face.h5')