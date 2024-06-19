import cv2
import numpy as np
import time
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

cap = cv2.VideoCapture(1)

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

fps_start_time = 0
fps = 0

while True:
    ret, image_org = cap.read()
    if not ret:
        continue

    fps_start_time = time.time()

    # Resize the image to 128x128 for prediction
    image_resized = cv2.resize(image_org, (128, 128))
    image = image_resized.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    class_index = np.argmax(predict[0])
    class_label = class_name[class_index]
    print(np.max(predict[0],axis=0))

    text = f"{class_label}"
    cv2.putText(image_org, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff

    cv2.putText(image_org,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,240,5),2)

    # Display the image with larger window size
    cv2.imshow('Camera', cv2.resize(image_org, (640, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
