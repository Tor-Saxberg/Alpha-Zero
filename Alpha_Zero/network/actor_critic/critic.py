from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = (6,7)
        self.action_size = (7,)

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=self.state_size, name='states')
        states_reshape = layers.Reshape((6,7,1))(states)
        actions = layers.Input(shape=self.action_size, name='actions')
        actions_reshape = layers.Reshape((7,1))(actions)

        # Add hidden layer(s) for state pathway
        for _ in range(3):
            net_states = layers.Conv2D(32,3)(states_reshape)
            net_states = layers.BatchNormalization()(net_states)
            net_states = layers.Activation("relu")(net_states)
        net_states = layers.Flatten()(net_states)
        net_states = layers.Dense(units=150)(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=150)(actions_reshape)
        net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        v_values = layers.Dense(units=1, name='q_values')(net)
        pi_values = layers.Dense(units=7, name='pi_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=[v_values, pi_values])

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        v_gradients = K.gradients(v_values, actions)
        pi_gradients= K.gradients(pi_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=[v_gradients, pi_gradients])
