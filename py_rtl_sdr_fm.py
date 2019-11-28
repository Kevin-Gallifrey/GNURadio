#!/usr/bin/env python3


import threading
import queue
import pyaudio
from scipy import signal
#from pylab import *
from rtlsdr import *
import numpy as np

class th(threading.Thread):
  def __init__(self, func):
    super(th, self).__init__()
    self.func = func

  def run(self):
    self.func()

def cb(samples,rtlobj):
  global q
  q.put(samples)

def sound():
  global qs
  global s
  while True:
    nd = qs.get()
    s.write(nd)


def defm():
  global q
  global qs
  while True:
    samples = q.get()
    delta = samples[0:-1] * samples[1:].conj()
    angs = np.angle(delta)
    nd = signal.decimate(angs, 25, ftype="fir")
    nd *= 20000
    qs.put(nd.astype(np.dtype('<i2')).tostring())

global q
global qs
global s

sdr = RtlSdr()

sdr.sample_rate = 1.2e6
sdr.center_freq = 97.5e6
sdr.gain = 49.6 

p = pyaudio.PyAudio()
s = p.open(format=p.get_format_from_width(2), channels=1, rate=48000, output=True)
q = queue.Queue()
qs = queue.Queue()
t1 = th(defm)
t1.start()
t = th(sound)
t.start()
sdr.read_samples_async(cb, 1024*500)

