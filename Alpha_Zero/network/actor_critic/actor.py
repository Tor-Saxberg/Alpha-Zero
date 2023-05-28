from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=self.state_size, name='states')
        states_reshape = layers.Reshape((6,7,1))(states)

        # Add hidden layers
        for _ in range(3):
            net_states = layers.Conv2D(64, 4, padding='same')(states_reshape)
            net_states = layers.BatchNormalization()(net_states)
            net_states = layers.Activation("relu")(net_states)
        net_states = layers.Flatten()(net_states)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='softmax', name='raw_actions')(net_states)
        v = layers.Dense(1, activation='sigmoid', name='v')(net_states)
        pi = layers.Dense(units=self.action_size, activation='sigmoid', name='pi_raw')(net_states)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=[v, pi])

        # Define loss function using action value (Q value) gradients
        v_gradients = layers.Input(shape=(self.action_size,))
        pi_gradients = layers.Input(shape=(self.action_size,))
        loss = K.sum((v-v_gradients)**2 -pi_gradients * K.log(pi))

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, v_gradients, pi_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
