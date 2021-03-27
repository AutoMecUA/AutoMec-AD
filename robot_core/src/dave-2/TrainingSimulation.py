from utils import *
from sklearn.model_selection import train_test_split
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Todo - Adicionar conditionals para treinar o modelo se ele já existe

# STEP 1 - Initialize Data
path = 'myData'
data = importDataInfo(path)

# Step 2 - Vizualize and Balance data
balanceData(data, display=True)

# Step 3 - Prepare for Processing
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0], steering[0])

# Step 4 - Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(
    imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))


# Step 5- Augmentation and Variation to the Data
# Defined Helper function

# Step 6 - Preprocessing

# Step 7 - Batch Generator

# Step 8 - Creating the Model
model = createModel()
model.summary()

# Step 9 -Training
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=200, epochs=3,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

# Step 10 - Saving and plotting
model.save('model.h5')
print('\n Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


# TODO EDITAR ISTO PARA PODER TREINAR MULTIPLAS VEZES
