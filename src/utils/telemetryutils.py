import numpy as np
import tensorflow as tf

#===========================================================

class TBSummary:

    def __init__(self):
        self.summary = []

    def add_value(self, name, value):
        self.summary.append(tf.Summary.Value(tag=name, simple_value=value))
        return self
\
#===========================================================

def outputStats(writer, it, loss, name='loss'):
    tbSum = TBSummary()
    tbSum.add_value(name, loss)
    writer.add_summary(tf.Summary(value=tbSum.summary), it)