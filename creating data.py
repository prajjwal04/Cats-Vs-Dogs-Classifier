#Importing Libraries
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os

#Importing Datasets
DataDir = r"C:\Users\prajj\Desktop\cats vs dogs\PetImages/"

CATEGORIES = ["Dog" , "Cat"]
"""
for i in CATEGORIES:
    path= os.path.join(DataDir,i)
    for img in os.listdir(path):
        img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        break
    break

#Resizing all the images by 100x100
img_size = 120
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap= 'gray')
plt.show()

training_data = []

def create_training_data():
    for i in CATEGORIES:
        path=os.path.join(DataDir,i)
        class_num = CATEGORIES.index(i)
        
 #Removing Corrupted file       
        for img in os.listdir(path):
            try:
                 img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                 new_array=cv2.resize(img_array,(img_size,img_size))
                 training_data.append([new_array,class_num])
            except Exception as e:
                pass
                      
create_training_data()
print(len(training_data))

#Shuffling the images
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].reshape(-1,img_size,img_size,1))

X = np.array(X).reshape(-1,img_size,img_size,1) #converting the array to numpy array
                
print(np.array(X).reshape(-1,img_size,img_size,1))    


import pickle

pickle_out=open(r"C:\Users\prajj\Desktop\cats vs dogs\X.pickle","wb")    
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open(r"C:\Users\prajj\Desktop\cats vs dogs\y.pickle","wb")   
pickle.dump(y,pickle_out)
pickle_out.close()
 
 """      














