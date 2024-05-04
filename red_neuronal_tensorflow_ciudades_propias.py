import numpy as np
import os
from scipy import stats

# TensorFlow
import tensorflow as tf
from keras import activations

print(tf.__version__)

def circulo(num_datos=100, R=1, minimo=0, maximo=1, latitud=0, longitud=0):
    pi = np.pi

    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x + longitud, 3)
    y = np.round(y + latitud, 3)

    df = np.column_stack([x, y])
    return df

N = 500

# Se utilizan puntos geográficos de las ciudades de Buenos Aires, Argentina y Barcelona, España
datos_buenosaires = circulo(num_datos=N, R=1.5, latitud=-34.61315, longitud=-58.37723)
datos_barcelona = circulo(num_datos=N, R=1, latitud=41.38879, longitud=2.15899)
X = np.concatenate([datos_buenosaires, datos_barcelona])
X = np.round(X, 3)
print ('X : ', X)

y = [0] * N + [1] * N
y = np.array(y).reshape(len(y), 1)
print ('y : ', y)

train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=4, input_shape=[2], activation=activations.relu, name='relu1'),
                                           tf.keras.layers.Dense(units=8, activation=activations.relu, name='relu2'),
                                           tf.keras.layers.Dense(units=1, activation=activations.sigmoid, name='sigmoid')])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500)
w = linear_model.layers[0].get_weights()[0]
b = linear_model.layers[0].get_weights()[1]
print('W 0', w)
print('b 0', b)
w = linear_model.layers[1].get_weights()[0]
b = linear_model.layers[1].get_weights()[1]
print('W 1', w)
print('b 1', b)
w = linear_model.layers[2].get_weights()[0]
b = linear_model.layers[2].get_weights()[1]
print('W 2', w)
print('b 2', b)

print('predict city 1 : Buenos Aires') # CAMBIAR COORDENADAS
buenosaires_matrix = tf.constant([[-34.6083, -58.3712],
                               [-34.6037, -58.3816],
                               [-34.6106, -58.3621],
                               [-34.5885, -58.3974],
                               [-34.6345, -58.3634],
                               [-34.5895, -58.4240],
                               [-34.5809, -58.4202],
                               [-34.5713, -58.4565],
                               [-34.5997, -58.3840],
                               [-34.4267, -58.5746]], tf.float32)
print(linear_model.predict(buenosaires_matrix).tolist())

print('predict city 2 : Barcelona') # CAMBIAR COORDENADAS
barcelona_matrix = tf.constant([[41.4036, 2.1744],
                                 [41.4147, 2.1526],
                                 [41.3809, 2.1739],
                                 [41.3834, 2.1765],
                                 [41.3809, 2.1228],
                                 [41.3645, 2.1552],
                                 [41.3918, 2.1649],
                                 [41.3781, 2.1886],
                                 [41.3936, 2.1634],
                                 [41.4217, 2.1185]], tf.float32)
print(linear_model.predict(barcelona_matrix).tolist())

#export_path = 'linear-model/1/'
#tf.saved_model.save(linear_model, os.path.join('./',export_path))