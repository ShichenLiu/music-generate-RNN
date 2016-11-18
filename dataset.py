import madmom
import random
import numpy as np

filename = 'data/Tc-ma_013.wav'
sent_dim = 20
batch_size = 50

class Dataset:
    def __init__(self, filename, sent_dim):
        self.signal, self.sample_rate = madmom.audio.signal.load_wave_file(filename)
        self.signal = madmom.audio.signal.rescale(self.signal)
        self.n_samples = self.signal.shape[0]
        self.sent_dim = sent_dim
        return
    def next_batch(self, batch_size):
        X_batch = []
        y_batch = []
        for i in xrange(batch_size):
            idx = random.randint(0, self.n_samples-self.sent_dim-2)
            X_batch.append(np.copy(self.signal[idx:idx+self.sent_dim]))
            y_batch.append(np.copy(self.signal[idx+1:idx+self.sent_dim+1]))
        return np.array(X_batch), np.array(y_batch)

def train():
    return Dataset(filename, sent_dim)
