from utils import *
from sklearn.model_selection import train_test_split
import os
import pathlib

from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Todo - Adicionar conditionals para treinar o modelo se ele já existe

# STEP 1 - Initialize Data
s = str(pathlib.Path(__file__).parent.absolute())
path = s + '/data/'
data = importDataInfo(path)


# Step 2 - Vizualize and Balance data
balanceData(data, display=True)

# Step 3 - Prepare for Processing
imagesPath, steerings = loadData(path, data)
print(imagesPath[0], steerings[0])

# Step 4 - Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(
    imagesPath, steerings, test_size=0.001, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))


# Step 5- Augmentation and Variation to the Data
# Defined Helper function

# Step 6 - Preprocessing

# Step 7 - Batch Generator

# Step 8 - Creating the Model
print("\n" + "Create a new model from scratch? [Y/N]")
if input().lower() == "y":
    model = createModel()
else:
    s = str(pathlib.Path(__file__).parent.absolute())
    path = s + '/models_files/model_daniel2.h5'
    model = load_model(path)


model.summary()

# Step 9 -Training
history = model.fit(batchGen(xTrain, yTrain, 20, 1), steps_per_epoch=100, epochs=10,
                    validation_data=batchGen(xVal, yVal, 20, 0), validation_steps=50)

# Step 10 - Saving and plotting
model.save('models_files/model_test2.h5')
print('\n Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


# TODO EDITAR ISTO PARA PODER TREINAR MULTIPLAS VEZES
