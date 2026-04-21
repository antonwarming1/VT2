#load model
from tensorflow import keras
import numpy as np

MODEL_SAVE_PATH = 'best_fnn_model.keras'

#test models on new data intances after loading the model
def load_model_and_predict(X_new):
    # Load the saved model
    loaded_model = keras.savings.load_model(MODEL_SAVE_PATH)
    print(f"Model loaded from {MODEL_SAVE_PATH}")
    
    # Make predictions on new data
    predictions = loaded_model.predict(X_new)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predicted_classes

