import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class_name = ['nen', 'xanh', 'vang', 'trang']

def get_model():  
    model_resnet50v2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    for layer in model_resnet50v2.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='image_input')
    output_resnet50v2 = model_resnet50v2(input)

    x = Flatten(name='flatten')(output_resnet50v2)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

my_model = get_model()
my_model.load_weights("resnet_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    prediction = my_model.predict(img_array)
    class_index = np.argmax(prediction[0])
    class_label = class_name[class_index]
    confidence = np.max(prediction[0])

    text = f"{class_label}: {confidence:.2f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Thay đổi đường dẫn tới file ảnh của bạn
img_path = 'data\chalk_blue.png'
predict_image(img_path)
