import pandas as pd
from sklearn.model_selection import train_test_split
import loss_function as lf
from tensorflow import keras
import matplotlib.pyplot as plt

columns = ["var", "skewness", "curtosis", "entropy", "class"]

data = pd.read_csv("data_banknote_authentication.txt", index_col=False, names=columns)

X = data.drop(columns=['class'])
y = data['class']

# Розділяємо дані на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Define loss functions and their names
loss_functions = {
    'Logistic Loss': lf.logistic_loss,
    'AdaBoost Loss': lf.adaboost_loss,
    'Binary Cross-Entropy': lf.binary_cross_entropy_loss
}

def train_and_get_loss(X_train, y_train, X_val, y_val, loss_function):
    model = keras.Sequential([
        keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_val, y_val))

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    return loss_train, loss_val, model

# Use map to train models with different loss functions
loss_trains, loss_vals, models = zip(*map(lambda item: train_and_get_loss(X_train, y_train, X_val, y_val, item[1]), loss_functions.items()))

# Plot training and validation loss curves
epochs = range(1, 51)
for loss_name, loss_train, loss_val in zip(loss_functions.keys(), loss_trains, loss_vals):
    plt.plot(epochs, loss_train, label=f'Training loss ({loss_name})')
    plt.plot(epochs, loss_val, label=f'Validation loss ({loss_name})', linestyle='dashed')

plt.title('Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(0.45, 0.5), loc='upper left', fontsize='small')
plt.grid()
plt.savefig('plot.png')

# Evaluate models on test data
for loss_name, model in zip(loss_functions.keys(), models):
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Accuracy using {loss_name}: {accuracy:.4f}")