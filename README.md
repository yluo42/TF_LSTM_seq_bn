# TF_LSTM_seq_bn

Sequential batch normalization for LSTM written in Tensorflow. Implemented
            according to *Cooijmans T, Ballas N, Laurent C, et al. Recurrent Batch Normalization, [arXiv:1603.09025](https://arxiv.org/pdf/1603.09025v4.pdf)*.

## Usage

```Python
import tensorflow as tf
from tf_lstm import LSTMCell

"""
Just simply replace tf.nn.rnn_cell.LSTMCell with LSTMCell
"""

deterministic = tf.Variable(False, name='deterministic) # when training, set to False; when testing, set to True

lstm = LSTMCell(n_hidden, use_peepholes=True, bn=True, deterministic=deterministic)
initial_state = lstm.zero_state(batch_size, tf.float32)
output, _states = tf.nn.rnn(lstm, input, initial_state=initial_state)

# training
session.run(...)

# when testing, set deterministic to True
session.run(tf.assign(deterministic, True))
# testing
session.run(...)
```
