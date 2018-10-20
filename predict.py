import numpy as np
from keras.preprocessing import image
from keras.models import load_model
classifier = load_model('./classifier.h5')
test_image = image.load_img('test_3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'sunflower'
else:
    prediction ='rose'
print(prediction)
