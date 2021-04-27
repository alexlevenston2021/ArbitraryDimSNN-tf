import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';
import warnings; warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning);

MAX_WEIGHT = 1.0
MIN_WEIGHT = -1.0

import tensorflow as tf;
import STDP_implementations as STDP;
import topology_generators as topologies;

def normalInitializer(net,mean=0.0,stddev=0.05):
    net.weights = tf.random.normal(net.weights.shape,mean=mean,stddev=stddev,dtype=net.weights.dtype)

class Network:
    def __init__(self, neuronDims, numSynapses, weightInitializer=normalInitializer, synapseInitializer=topologies.uniform, float_type=tf.float32, int_type=tf.int32):
        self.neuronDims = list(neuronDims)
        self.numSynapses = int(numSynapses)
        self.weightInitializer = weightInitializer
        self.synapseInitializer = synapseInitializer
        self.float_type = float_type
        self.int_type = int_type
        self.membranePotentials = tf.ones(self.neuronDims+[1], dtype=self.float_type)
        self.spikeTimes = tf.zeros(self.neuronDims+[1], dtype=self.float_type)
        self.weights = tf.zeros(self.neuronDims+[self.numSynapses,1],dtype=self.float_type)
        self.weightInitializer(self)
        self.synapseIndices = tf.zeros(self.neuronDims+[self.numSynapses, len(self.neuronDims)], dtype=self.int_type)
        self.synapseInitializer(self)
    def __call__(self, currentTime, lr=.01, inputs=None, threshold=30.0, resting=-70.0, time_step=1.0, tau=15.0, amplitude=2.0):
        if inputs is not None:
            self.membranePotentials += inputs
        aboveThreshold = tf.math.greater_equal(self.membranePotentials, threshold)
        self.membranePotentials = tf.where(aboveThreshold,
                                            resting,
                                            self.membranePotentials * tf.math.exp(-time_step/tau)
                                            ) + tf.reduce_sum(self.weights * tf.gather_nd(tf.cast(aboveThreshold, self.float_type)*threshold,self.synapseIndices),
                                                            axis=-2
                                                            )
        aboveThreshold = tf.math.greater_equal(self.membranePotentials, threshold)
        self.spikeTimes = tf.where(aboveThreshold, currentTime, self.spikeTimes)
        if lr != 0.0:
            self.weights += tf.where(tf.reshape(aboveThreshold,list(aboveThreshold.shape)+[1]), tf.math.sign(self.weights) * lr * tf.gather_nd(STDP.basicSTDP(currentTime,self.spikeTimes,tau,amplitude),self.synapseIndices) * (MAX_WEIGHT - self.weights) * (self.weights - MIN_WEIGHT), 0)
            self.spikeTimes = tf.where(aboveThreshold, 0, self.spikeTimes)
