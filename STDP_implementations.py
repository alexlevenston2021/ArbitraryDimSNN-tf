import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';
import warnings; warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning);

import tensorflow as tf

def basicSTDP(currentTime, lastFiring, tau, amplitude): #currentTime is a scalar, lastFiring is a tensor of shape [neuronDims] + [1], tau is STDP time decay constant, amplitude is the largest possible return value
    return tf.where(tf.math.logical_and(tf.math.greater_equal(currentTime, lastFiring), tf.math.not_equal(lastFiring, 0)), #currentTime is greater than or equal to the time of the last recorded unhandled presynaptic action potential, and the time of the last recorded unhandled presynaptic action potential is not zero (which would indicate it was handled, or that there has yet to be an action potential injected into that synapse)
                    2 * amplitude * (tf.math.exp(-(currentTime - lastFiring) / tau) - 0.5), #if last presynaptic firing has yet to be handled by learning rule, weight update follows STDP rule
                    0   #if no record of last presynaptic firing, weight update is 0
                    )

