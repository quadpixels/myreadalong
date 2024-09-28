import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Layer
from tensorflow.keras.models import Model, Sequential

# Example on https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07

# 1. Define model
my_model = Sequential([
    Input(name="the_inputs", shape=(5, 5, 1)),
    Conv2D(name="conv2d_1", filters=1, kernel_size=(3,3), activation="linear")
])
my_model.summary()
# 1.1. Manually assign weight
w = my_model.get_layer("conv2d_1").get_weights()
my_w = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])
my_w = my_w.reshape((3,3,1,1))
my_model.get_layer("conv2d_1").set_weights([my_w, np.array([0])])

# 2. Feed example input data
in0 = np.array([
    [7,2,3,3,8],
    [4,5,3,8,4],
    [3,3,2,8,4],
    [2,8,7,2,7],
    [5,4,4,5,4]
])

in0 = in0.reshape((1,5,5,1))  # Add batch dimension and channel dimension
out0 = my_model.predict(in0)
out0 = out0.reshape((3,3))

# 3. Save model to HDF5
my_model.save_weights("weights.h5", overwrite=True)