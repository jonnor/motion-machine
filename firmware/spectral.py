
import math
import emlearn_fft
import array

class SpectralFeaturesExtractor:
    def __init__(self, window_length, dimensions=3,
        fft_start=0, fft_end=16, sample_rate=50, sample_scale=2**15):
        
        self.window_length = window_length
        self.dimensions = dimensions
        self.fft_start = fft_start
        self.fft_end = fft_end
        self.sample_rate = sample_rate
        self.sample_scale = sample_scale

        fft_length = self.window_length
        self.fft = emlearn_fft.FFT(fft_length)
        emlearn_fft.fill(self.fft, fft_length)
        self.fft_real = array.array('f', (0 for _ in range(fft_length)))
        self.fft_imag = array.array('f', (0 for _ in range(fft_length)))
        self.spectrum = array.array('f', (0 for _ in range(fft_length // 2)))

    def compute_spectrum(self, samples):
        """Run FFT on magnitude signal, populates self.spectrum."""
        samples_length = len(samples) // self.dimensions
        scale = self.sample_scale

        for i in range(samples_length):
            x = samples[(i*3)+0] / scale
            y = samples[(i*3)+1] / scale
            z = samples[(i*3)+2] / scale
            self.fft_real[i] = x*x + y*y + z*z
            self.fft_imag[i] = 0

        self.fft.run(self.fft_real, self.fft_imag)

        for i in range(len(self.spectrum)):
            r = self.fft_real[i]
            im = self.fft_imag[i]
            self.spectrum[i] = (r*r + im*im) ** 0.5

    def _freq_to_bin(self, freq_hz):
        return int(freq_hz * self.window_length / self.sample_rate)


    def spectral_energy(self, start_bin=None, end_bin=None, spectrum=None):
        sp = self.spectrum if spectrum is None else spectrum
        s = self.fft_start if start_bin is None else start_bin
        e = self.fft_end   if end_bin   is None else end_bin
        energy = 0.0
        for i in range(s, e):
            energy += sp[i] * sp[i]
        return energy

    def subband_energy(self, freq_low, freq_high, spectrum=None):
        return self.spectral_energy(self._freq_to_bin(freq_low), self._freq_to_bin(freq_high), spectrum)

    def dominant_frequency(self, start_bin=None, end_bin=None, spectrum=None):
        sp = self.spectrum if spectrum is None else spectrum
        s = self.fft_start if start_bin is None else start_bin
        e = self.fft_end   if end_bin   is None else end_bin
        peak_bin = s
        peak_val = sp[s]
        for i in range(s + 1, e):
            if sp[i] > peak_val:
                peak_val = sp[i]
                peak_bin = i
        return peak_bin * self.sample_rate / self.window_length

    def spectral_centroid(self, start_bin=None, end_bin=None, spectrum=None):
        sp = self.spectrum if spectrum is None else spectrum
        s = self.fft_start if start_bin is None else start_bin
        e = self.fft_end   if end_bin   is None else end_bin
        weighted_sum = 0.0
        total = 0.0
        for i in range(s, e):
            weighted_sum += i * sp[i]
            total += sp[i]
        if total < 1e-10:
            return 0.0
        return (weighted_sum / total) * self.sample_rate / self.window_length

    def spectral_spread(self, start_bin=None, end_bin=None, spectrum=None):
        sp = self.spectrum if spectrum is None else spectrum
        s = self.fft_start if start_bin is None else start_bin
        e = self.fft_end   if end_bin   is None else end_bin
        centroid_bin = self.spectral_centroid(s, e, sp) * self.window_length / self.sample_rate
        weighted_sq = 0.0
        total = 0.0
        for i in range(s, e):
            diff = i - centroid_bin
            weighted_sq += sp[i] * diff * diff
            total += sp[i]
        if total < 1e-10:
            return 0.0
        return (weighted_sq / total) ** 0.5 * self.sample_rate / self.window_length

    def spectral_entropy(self, start_bin=None, end_bin=None, spectrum=None):
        sp = self.spectrum if spectrum is None else spectrum
        s = self.fft_start if start_bin is None else start_bin
        e = self.fft_end   if end_bin   is None else end_bin
        total_energy = 0.0
        for i in range(s, e):
            total_energy += sp[i] * sp[i]
        if total_energy < 1e-10:
            return 0.0
        entropy = 0.0
        for i in range(s, e):
            p = (sp[i] * sp[i]) / total_energy
            if p > 1e-10:
                entropy -= p * (math.log(p) / math.log(2))
        return entropy

    def preprocess(self, samples: array.array):
        assert len(samples) == (self.window_length * self.dimensions)
        self._compute_raw_fft(samples)

        fft_energy = sum(self.spectrum)
        scale = 2**14 / fft_energy if fft_energy > 1e-6 else 0.0

        return [self.spectrum[i] * scale for i in range(self.fft_start, self.fft_end)]

    def extract_all(self, samples: array.array):
        """Return spectral feature vector. Call after preprocess(), or pass samples to recompute."""
        assert len(samples) == (self.window_length * self.dimensions)
        self._compute_raw_fft(samples)

        return [
            self.spectral_energy(),
            self.dominant_frequency(),
            self.spectral_centroid(),
            self.spectral_spread(),
            self.spectral_entropy(),
        ]
