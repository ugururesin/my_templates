import tensorflow as tf
import tensorflow_federated as tff

# Define a simple model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')
    return model

# Define the federated data
train_data, _ = tff.simulation.datasets.synthetic.noise(
    num_clients=10,
    num_examples=1000,
    noise_multiplier=0.1)
    
# Wrap the data in a federated dataset
train_data = train_data.preprocess(lambda x: (x['x'], x['y']))
train_data = tff.simulation.client_data.ConcreteClientData({i: train_data.create_tf_dataset_for_client(i) for i in range(10)})
train_data = tff.simulation.datasets.TestClientData(train_data)

# Define the Federated Averaging process
@tff.federated_computation
def server_init():
    return tff.learning.from_keras_model(create_model())

@tff.federated_computation
def client_update(model, dataset):
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    keras_model = tff.learning.from_keras_model(model)
    keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss=loss_fn)
    keras_model.fit(dataset)
    return keras_model.weights.trainable

@tff.federated_computation
def server_update(model, mean_client_weights):
    new_weights = []
    weight_sum = 0.0
    for client_weight in mean_client_weights:
        new_weights.append(client_weight)
        weight_sum += 1.0
    for model_weight in model.weights.trainable:
        new_weight_value = 0.0
        for client_weight in mean_client_weights:
            new_weight_value += client_weight[model_weight.name] / weight_sum
        new_weights.append(new_weight_value)
    new_model = tff.learning.from_keras_model(create_model())
    new_model.weights = new_weights
    return new_model

# Run the Federated Averaging process
iterative_process = tff.learning.build_federated_averaging_process(server_init, client_update, server_update)
state = iterative_process.initialize()
for i in range(10):
    state, metrics = iterative_process.next(state, train_data)
    print('Round {}: loss={}'.format(i, metrics.loss))
