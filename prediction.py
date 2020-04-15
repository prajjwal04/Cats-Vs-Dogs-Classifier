import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model 

CATEGORIES = ['Doogo', 'Kitties']  # will use this to convert prediction num to string value
image = r'C:\Users\prajj\Desktop\cats vs dogs\Test Images\11.jpg'

def prepare(image):
    img_size =80
    img_array= cv2.imread(image,cv2.IMREAD_GRAYSCALE) #read in the image,convert to grayscale
    new_array=cv2.resize(img_array,(img_size,img_size))  #resize image to match models expected sizing
    return new_array.reshape(-1,img_size,img_size,1) #return the image with shaping that TF wants


model = tf.keras.models.load_model(r'C:\Users\prajj\Desktop\cats vs dogs\64x3-CNN.model')
prediction = model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()