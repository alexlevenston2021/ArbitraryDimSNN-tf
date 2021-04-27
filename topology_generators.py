import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';
import warnings; warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning);

import tensorflow as tf

def uniform(net):
    net.synapseIndices = tf.reshape(tf.concat([tf.random.uniform(net.synapseIndices.shape[:-1],0,net.neuronDims[i],dtype=net.synapseIndices.dtype) for i in range(len(net.neuronDims))], axis=-1),
                        net.synapseIndices.shape
                        )

