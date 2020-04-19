from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 

model = load_model('/Users/matrikasubedi/Documents/10-monkey-species/monkey_breed_mobilNet.h5')

img_rows,img_cols = 224,224

class_labels = [
	'mantled_howler', 
	'patas_monkey', 
	'bald_uakari', 
	'japanese_macaque', 
	'pygmy_marmoset', 
	'white_headed_capuchin',
	'silvery_marmoset',
	'common_squirrel_monkey', 
	'black_headed_night_monkey',
	'nilgiri_langur' 
	]
def check(path):
    
    
    # prediction
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255
    pred = np.argmax(model.predict(x))
   
    print("It's a {}.".format(class_labels[pred])) 
  
check('/Users/matrikasubedi/Documents/10-monkey-species/test images/black_headed_night_monkey.jpg')