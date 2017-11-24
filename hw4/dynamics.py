import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ Note: Be careful about normalization """

        self.epsilon = 1e-8

        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess

        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name='inputs_obs')
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_act], name='inputs_action')
        self.obs_tp1_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name='inputs_obs_next')

        obs_norm = (self.obs_ph - self.mean_obs) / (self.std_obs + self.epsilon)
        actions_norm = (self.action_ph - self.mean_action) / (self.std_action + self.epsilon)

        self.inputs = tf.concat((obs_norm, actions_norm), 1, name='inputs')

        self.obs_diff_norm_predicted = build_mlp(
            self.inputs,
            n_obs,
            'dynamics',
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation,
        )

        self.obs_diff_norm_targets = ((self.obs_tp1_ph - self.obs_ph) - self.mean_deltas) / (self.std_deltas + self.epsilon)

        self.loss = tf.losses.mean_squared_error(self.obs_diff_norm_targets, self.obs_diff_norm_predicted)
        self.update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        for i in range(self.iterations):
            loss_iter = 0
            for data_batch in batch(data, self.batch_size):
                feed = {
                    self.obs_ph: data_batch['observations'],
                    self.action_ph: data_batch['actions'],
                    self.obs_tp1_ph: data_batch['next_observations'],
                }
                loss, _ = self.sess.run([self.loss, self.update], feed_dict=feed)
                loss_iter += loss * data_batch['observations'].shape[0]
            loss_iter /= data['observations'].shape[0]
            logger.info('Dynamics model iter %i loss %f', i, loss_iter)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """

        feed = {
            self.obs_ph: states,
            self.action_ph: actions,
        }

        obs_deltas_normalized = self.sess.run(self.obs_diff_norm_predicted, feed_dict=feed)

        obs_deltas = obs_deltas_normalized * self.std_deltas + self.mean_deltas

        obs_tp1 = states + obs_deltas

        return obs_tp1


def batch(data, batch_size):

    n = data['observations'].shape[0]

    slice_batch = slice(0, batch_size)

    while slice_batch.start < n:
        data_batch = {k: v[slice_batch] for k, v in data.items()}

        yield data_batch

        slice_batch = slice(slice_batch.stop, slice_batch.stop + batch_size)
