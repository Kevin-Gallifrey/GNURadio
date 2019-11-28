#!/usr/bin/env python

import pyaudio
from scipy import signal
from rtlsdr import *
import numpy as np


p = pyaudio.PyAudio()
s = p.open(format=p.get_format_from_width(2), channels=1, rate=48000, output=True)

sdr = RtlSdr()

# configure device
sdr.sample_rate = 960e3
sdr.center_freq = 97.5e6
sdr.gain = 49

samples = sdr.read_samples(512*14000)

delta = samples[0:-1] * samples[1:].conj()
angs = np.angle(delta)

nd = signal.decimate(angs, 20, ftype="fir")

nd *= 20000

s.write(nd.astype(np.dtype('<i2')).tostring())
