import array
import math
import sys

from spectral import SpectralFeaturesExtractor

# --- Helpers ---

def make_samples(window_length, dimensions, freq_hz, sample_rate, amplitude=1.0):
    """Single-axis sinusoid at freq_hz, other axes zero."""
    samples = array.array('h', (0 for _ in range(window_length * dimensions)))
    for i in range(window_length):
        val = int(amplitude * 2**15 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        samples[i * dimensions + 0] = val
        samples[i * dimensions + 1] = 0
        samples[i * dimensions + 2] = 0
    return samples

def make_dc_samples(window_length, dimensions, amplitude=1.0):
    """Constant (DC) signal on x axis."""
    samples = array.array('h', (0 for _ in range(window_length * dimensions)))
    val = int(amplitude * 2**15)
    for i in range(window_length):
        samples[i * dimensions + 0] = val
        samples[i * dimensions + 1] = 0
        samples[i * dimensions + 2] = 0
    return samples

def assert_close(name, actual, expected, tol=0.05):
    err = abs(actual - expected) / (abs(expected) + 1e-10)
    assert err < tol, f"{name}: expected {expected:.4f}, got {actual:.4f} (rel err {err:.3f})"


WINDOW = 64
DIMS = 3
SR = 50


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

def test_dominant_freq_bin2():
    freq = 2 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), freq, tol=0.01)

def test_dominant_freq_bin4():
    freq = 4 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), freq, tol=0.01)

def test_dominant_freq_bin6():
    freq = 6 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    assert_close("dominant_freq", ext.dominant_frequency(), freq, tol=0.01)

def test_centroid_bin3():
    freq = 3 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    assert_close("centroid", ext.spectral_centroid(), freq, tol=0.05)

def test_centroid_bin5():
    freq = 5 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    assert_close("centroid", ext.spectral_centroid(), freq, tol=0.05)

def test_subband_energy_inband():
    target_bin = 8
    freq = target_bin * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    bin_width = SR / WINDOW
    energy_in  = ext.subband_energy(freq - bin_width, freq + bin_width)
    energy_out = ext.subband_energy(0.0, freq - 2 * bin_width)
    assert energy_in > energy_out * 10, \
        f"in-band={energy_in:.4f} should dwarf out-of-band={energy_out:.4f}"

def test_subband_energy_external_spectrum():
    """Passing an external spectrum array should give same result as self.spectrum."""
    target_bin = 8
    freq = target_bin * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    bin_width = SR / WINDOW
    # Copy spectrum to a separate array and pass explicitly
    external = array.array('f', ext.spectrum)
    e1 = ext.subband_energy(freq - bin_width, freq + bin_width)
    e2 = ext.subband_energy(freq - bin_width, freq + bin_width, spectrum=external)
    assert_close("subband_energy external", e1, e2, tol=1e-5)

def test_entropy_tone_low():
    freq = 4 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    entropy = ext.spectral_entropy()
    max_entropy = math.log(WINDOW // 2) / math.log(2)
    assert entropy < max_entropy * 0.5, \
        f"tone entropy={entropy:.3f} should be well below max={max_entropy:.3f}"

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
    max_entropy = math.log(WINDOW // 2) / math.log(2)
    assert entropy > max_entropy * 0.5, \
        f"noise entropy={entropy:.3f} should be above half of max={max_entropy:.3f}"

def test_entropy_noise_greater_than_tone():
    import random
    freq = 4 * SR / WINDOW
    ext_tone = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext_tone.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))

    random.seed(42)
    ext_noise = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    samples_noise = array.array('h', (
        random.randint(-2**14, 2**14) if i % DIMS == 0 else 0
        for i in range(WINDOW * DIMS)
    ))
    ext_noise.compute_spectrum(samples_noise)

    assert ext_noise.spectral_entropy() > ext_tone.spectral_entropy() * 2, \
        f"noise entropy={ext_noise.spectral_entropy():.3f} should be >> tone={ext_tone.spectral_entropy():.3f}"

def test_external_spectrum_dominant_freq():
    """dominant_frequency() with an external spectrum should use that spectrum."""
    freq = 5 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    external = array.array('f', ext.spectrum)
    # Corrupt self.spectrum so we can tell which one was used
    for i in range(len(ext.spectrum)):
        ext.spectrum[i] = 0.0
    dom = ext.dominant_frequency(spectrum=external)
    assert_close("external dominant_freq", dom, freq, tol=0.01)

def test_external_spectrum_centroid():
    """spectral_centroid() with an external spectrum should use that spectrum."""
    freq = 3 * SR / WINDOW
    ext = SpectralFeaturesExtractor(WINDOW, DIMS, fft_start=0, fft_end=WINDOW//2, sample_rate=SR)
    ext.compute_spectrum(make_samples(WINDOW, DIMS, freq, SR, amplitude=0.8))
    external = array.array('f', ext.spectrum)
    for i in range(len(ext.spectrum)):
        ext.spectrum[i] = 0.0
    centroid = ext.spectral_centroid(spectrum=external)
    assert_close("external centroid", centroid, freq, tol=0.05)


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
