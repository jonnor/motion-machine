import array
import math
import sys

from spectral import SpectralFeaturesExtractor


# --- Helpers ---

def make_samples(window_length, dimensions, freq_hz, sample_rate, amplitude=1.0):
    """
    Single-axis sinusoid at freq_hz on x-axis, y/z zero.
    NOTE: the FFT input is x^2+y^2+z^2 = sin^2, which by the identity
    sin^2(w) = (1 - cos(2w))/2 produces:
      - a DC component at bin 0
      - a tone at 2*freq_hz (i.e. double the input frequency bin)
    Tests must use the *doubled* frequency as the expected dominant/centroid.
    """
    samples = array.array('h', (0 for _ in range(window_length * dimensions)))
    for i in range(window_length):
        val = int(amplitude * 2**15 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        samples[i * dimensions + 0] = val
        samples[i * dimensions + 1] = 0
        samples[i * dimensions + 2] = 0
    return samples

def make_dc_samples(window_length, dimensions, amplitude=1.0):
    """Constant signal on x-axis — produces pure DC (bin 0) after squaring."""
    samples = array.array('h', (0 for _ in range(window_length * dimensions)))
    val = int(amplitude * 2**15)
    for i in range(window_length):
        samples[i * dimensions + 0] = val
        samples[i * dimensions + 1] = 0
        samples[i * dimensions + 2] = 0
    return samples

def assert_close(name, actual, expected, tol=0.05):
    err = abs(actual - expected) / (abs(expected) + 1e-10)
    assert err < tol, \
        name + ": expected " + str(round(expected, 4)) + \
        ", got " + str(round(actual, 4)) + \
        " (rel err " + str(round(err, 3)) + ")"


WINDOW = 64
DIMS   = 3
SR     = 50
BW     = SR / WINDOW  # bin width in Hz = 0.78125 Hz


# --- DC tests ---
# Constant input -> magnitude is constant -> all energy in bin 0.
# dominant_freq, centroid, spread, entropy are all 0.

def test_dc_dominant_freq():
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_dc_samples(WINDOW, DIMS, amplitude=0.5))
    assert_close("dominant_freq", ext.dominant_frequency(), 0.0)

def test_dc_centroid():
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_dc_samples(WINDOW, DIMS, amplitude=0.5))
    assert_close("centroid", ext.spectral_centroid(), 0.0)

def test_dc_spread():
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_dc_samples(WINDOW, DIMS, amplitude=0.5))
    assert_close("spread", ext.spectral_spread(), 0.0)

def test_dc_entropy():
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_dc_samples(WINDOW, DIMS, amplitude=0.5))
    assert_close("entropy", ext.spectral_entropy(), 0.0)


# --- Dominant frequency tests ---
# sin^2(2pi * f * t) = (1 - cos(2pi * 2f * t)) / 2
# So a sinusoid at bin k produces a peak at bin 2k.
# We use fft_start=1 to exclude the DC bin so dominant_frequency finds the tone.

def test_dominant_freq_bin2():
    input_freq = 2 * BW          # bin 2
    expected   = 2 * input_freq  # peak lands at bin 4
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), expected, tol=0.01)

def test_dominant_freq_bin4():
    input_freq = 4 * BW          # bin 4
    expected   = 2 * input_freq  # peak lands at bin 8
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), expected, tol=0.01)

def test_dominant_freq_bin6():
    input_freq = 6 * BW          # bin 6
    expected   = 2 * input_freq  # peak lands at bin 12
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), expected, tol=0.01)


# --- Centroid tests ---
# With DC included the centroid is pulled toward 0, so we exclude bin 0
# (fft_start=1) and expect the centroid near 2*input_freq (the tone bin).

def test_centroid_bin3():
    input_freq = 3 * BW
    expected   = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    assert_close("centroid", ext.spectral_centroid(), expected, tol=0.05)

def test_centroid_bin5():
    input_freq = 5 * BW
    expected   = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    assert_close("centroid", ext.spectral_centroid(), expected, tol=0.05)


# --- Subband energy test ---
# The tone lands at 2*input_freq. A narrow band around it should contain
# far more energy than a band well away from it.

def test_subband_energy_inband():
    input_freq  = 8 * BW          # bin 8 in -> bin 16 out
    tone_freq   = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    energy_in  = ext.subband_energy(tone_freq - BW, tone_freq + BW)
    # choose an out-of-band region well away from both DC and the tone
    energy_out = ext.subband_energy(tone_freq + 3*BW, tone_freq + 8*BW)
    assert energy_in > energy_out * 10, \
        "in-band=" + str(round(energy_in, 4)) + \
        " should dwarf out-of-band=" + str(round(energy_out, 4))

def test_subband_energy_external_spectrum():
    """Passing an external spectrum gives same result as self.spectrum."""
    input_freq = 8 * BW
    tone_freq  = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    external = array.array('f', ext.spectrum)
    e1 = ext.subband_energy(tone_freq - BW, tone_freq + BW)
    e2 = ext.subband_energy(tone_freq - BW, tone_freq + BW, spectrum=external)
    assert_close("subband external", e1, e2, tol=1e-5)


# --- Entropy tests ---
# A pure tone (after squaring: DC + one harmonic) has low entropy.
# Random noise spread across all bins has higher entropy.
# We use a relative comparison and a modest absolute threshold that
# accounts for single-axis noise (not perfectly flat spectrum).

def test_entropy_tone_low():
    input_freq = 4 * BW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    entropy = ext.spectral_entropy()
    max_entropy = math.log(WINDOW // 2) / math.log(2)
    assert entropy < max_entropy * 0.4, \
        "tone entropy=" + str(round(entropy, 3)) + \
        " should be well below max=" + str(round(max_entropy, 3))

def test_entropy_noise_high():
    import random
    random.seed(42)
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    samples = array.array('h', (
        random.randint(-2**14, 2**14) if i % DIMS == 0 else 0
        for i in range(WINDOW * DIMS)
    ))
    ext.compute_spectrum(samples)
    entropy = ext.spectral_entropy()
    # Single-axis noise isn't perfectly flat so use 0.4 not 0.5 as lower bound
    max_entropy = math.log(WINDOW // 2) / math.log(2)
    assert entropy > max_entropy * 0.4, \
        "noise entropy=" + str(round(entropy, 3)) + \
        " should be above 0.4 * max=" + str(round(max_entropy, 3))

def test_entropy_noise_greater_than_tone():
    import random
    input_freq = 4 * BW
    ext_tone = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext_tone.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))

    random.seed(42)
    ext_noise = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    samples_noise = array.array('h', (
        random.randint(-2**14, 2**14) if i % DIMS == 0 else 0
        for i in range(WINDOW * DIMS)
    ))
    ext_noise.compute_spectrum(samples_noise)

    assert ext_noise.spectral_entropy() > ext_tone.spectral_entropy() * 2, \
        "noise entropy=" + str(round(ext_noise.spectral_entropy(), 3)) + \
        " should be >> tone=" + str(round(ext_tone.spectral_entropy(), 3))


# --- External spectrum tests ---

def test_external_spectrum_dominant_freq():
    """dominant_frequency() uses the external spectrum, not self.spectrum."""
    input_freq = 5 * BW
    expected   = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    external = array.array('f', ext.spectrum)
    for i in range(len(ext.spectrum)):
        ext.spectrum[i] = 0.0
    assert_close("external dominant_freq", ext.dominant_frequency(spectrum=external), expected, tol=0.01)

def test_external_spectrum_centroid():
    """spectral_centroid() uses the external spectrum, not self.spectrum."""
    input_freq = 3 * BW
    expected   = 2 * input_freq
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=1, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, input_freq, SR, amplitude=0.8))
    external = array.array('f', ext.spectrum)
    for i in range(len(ext.spectrum)):
        ext.spectrum[i] = 0.0
    assert_close("external centroid", ext.spectral_centroid(spectrum=external), expected, tol=0.05)


# --- Test runner ---

def _collect_tests():
    g = globals()
    tests = [(name, g[name]) for name in g if name.startswith('test_')]
    tests.sort(key=lambda t: t[0])
    return tests

def run_all():
    tests = _collect_tests()
    passed = 0
    failed = 0
    failures = []
    print("test_spectral.py  (" + str(len(tests)) + " tests)")
    print("-" * 52)
    for name, fn in tests:
        try:
            fn()
            print("  PASS  " + name)
            passed += 1
        except AssertionError as e:
            msg = str(e) if str(e) else "AssertionError"
            print("  FAIL  " + name + "  -> " + msg)
            failures.append((name, msg))
            failed += 1
        except Exception as e:
            msg = type(e).__name__ + ": " + str(e)
            print("  ERROR " + name + "  -> " + msg)
            failures.append((name, msg))
            failed += 1
    print("-" * 52)
    print("Results: " + str(passed) + " passed, " + str(failed) + " failed")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print("  " + name + ": " + msg)
    return failed == 0

if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
