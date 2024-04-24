import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


input_shape = (150, 150, 3)

def create_model(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

num_classes = 2

model = create_model(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "F:/pythonProject/face_detection/detection",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        "F:/pythonProject/face_detection/valid",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

model.fit(
      train_generator,
      steps_per_epoch=int(train_generator.samples/train_generator.batch_size),
      validation_data=validation_generator,
      validation_steps=int(validation_generator.samples/validation_generator.batch_size),
      epochs=20)

model.save("facial_recognition_model.keras")