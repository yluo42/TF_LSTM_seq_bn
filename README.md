# TF_LSTM_seq_bn

Sequential batch normalization for LSTM written in Tensorflow. Implemented
            according to *Cooijmans T, Ballas N, Laurent C, et al. Recurrent Batch Normalization, [arXiv:1603.09025](https://arxiv.org/pdf/1603.09025v4.pdf)*.

In `LSTMCell` class, set `bn` to 1/2/3 to open batch normalization. Default is `bn=0`. Set `bn=1` only apply batch norm on `WX`, `bn=2` to apply batch norm on both `WX` and `Wh`, `bn=3` to apply batch norm on `WX`, `Wh` and `c`.

Be careful - set `bn` larger than 1 might be extremely slow in deep LSTM models!

## Usage

```Python
import tensorflow as tf
from tf_lstm import LSTMCell

"""
Just simply replace tf.nn.rnn_cell.LSTMCell with LSTMCell
"""

deterministic = tf.Variable(False, name='deterministic) # when training, set to False; when testing, set to True

lstm = LSTMCell(n_hidden, use_peepholes=True, bn=1, deterministic=deterministic) # level-1 batch norm
initial_state = lstm.zero_state(batch_size, tf.float32)
output, _states = tf.nn.rnn(lstm, input, initial_state=initial_state)

# training
session.run(...)

# when testing, set deterministic to True
session.run(tf.assign(deterministic, True))
# testing
session.run(...)
```
