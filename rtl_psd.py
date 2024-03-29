#!/usr/bin/env python

from pylab import *
from rtlsdr import *



sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 97.5e6
sdr.gain = 40

samples = sdr.read_samples(256*1024)

# use matplotlib to estimate and plot the PSD
psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')

show()

