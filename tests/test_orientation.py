
import sys
import array
import math
import random

from orientation import GravityEstimatorLowpass, normalize_gravity, compute_tilt

# ----
# Helpers
# -----

def assert_close_arr(a, b, tol=1e-3, msg=""):
    if len(a) != len(b):
        raise AssertionError("length mismatch " + str(len(a)) + " vs " + str(len(b)))
    for i in range(len(a)):
        if abs(a[i] - b[i]) > tol:
            raise AssertionError(
                msg or ("index " + str(i) + ": " + str(a[i]) + " vs " + str(b[i]))
            )


def simulate_accelerometer(duration,
        samplerate=25,
        noise_std=0.02,
        roll_rate=0.3,
        pitch_rate=0.2,
    ):
    """
    Simulates slow device rotation with accelerometer noise.
    Yields (t, [ax, ay, az]) samples.
    """
    dt = 1.0 / samplerate
    t = 0.0
    samples = int(duration*samplerate)

    for i in range(samples):
        roll  = roll_rate  * t
        pitch = pitch_rate * t

        # Gravity vector in device frame (unit vector pointing "down")
        ax = -math.sin(pitch)
        ay =  math.cos(pitch) * math.sin(roll)
        az =  math.cos(pitch) * math.cos(roll)

        # Add gaussian noise (Box-Muller)
        def gauss():
            u1 = random.random() or 1e-10
            u2 = random.random()
            return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        ax += gauss() * noise_std
        ay += gauss() * noise_std
        az += gauss() * noise_std

        yield t, [ax, ay, az]
        t += dt

# -----
# Tests
# -----
def test_gravity_estimator():

    import npyfile

    # From tools/orientation.py
    filter = {
        'samplerate': 25.0,
        'coefficients': [1.32937288987529e-05, 2.65874577975058e-05, 1.32937288987529e-05, 1.0, -1.778313488139435, 0.7924474718329468,
                         1.0, 2.0, 1.0, 1.0, -1.8934156010225003, 0.9084644129492953],
        'order': 4,
        'cutoff': 0.5,
    }

    coefficients = array.array('f', filter['coefficients'])
    est = GravityEstimatorLowpass(coefficients)

    duration = 20.0
    samplerate = 25
    accelerometer_stream = simulate_accelerometer(\
        duration=duration,
        samplerate=samplerate,
        noise_std=0.20,
        roll_rate=0.4,
        pitch_rate=0.3,
    )
    expect_samples = int(duration*samplerate)

    # Log the data so that it can be checked/visualized
    out_columns = [
        'time',
        'acc_x', 'acc_y', 'acc_z',
        'orientation_x', 'orientation_y', 'orientation_z',
        'pitch', 'roll',
    ]
    out = array.array('f', (0.0 for _ in range(len(out_columns))))
    out_typecode = 'f'
    out_shape = (expect_samples, len(out))
    output_path = 'test_gravity_estimator_rotations.npy'
    with npyfile.Writer(output_path,
                        shape=out_shape,
                        typecode=out_typecode) as outfile:

        sample = 0
        for t, xyz in accelerometer_stream:
            sample += 1
            gravity = est.update(xyz)
            norm = normalize_gravity(gravity)
            pitch, roll = compute_tilt(gravity)

            # XXX: careful to match column order
            out[0] = t
            out[1] = xyz[0]
            out[2] = xyz[1]
            out[3] = xyz[2]
            out[4] = norm[0]
            out[5] = norm[1]
            out[6] = norm[2]
            out[7] = pitch
            out[8] = roll
            outfile.write_values(out, typecode=out_typecode)

    assert sample == expect_samples, (sample, expect_samples)

    # TODO: check that pitch/roll spans entire range

def test_normalize_gravity_straight():

    # with list
    xyz = [9.8, 0.0, 0.0]
    norm = normalize_gravity(xyz)
    assert_close_arr(norm, [1.0, 0.0, 0.0])

    # with float array
    xyz = array.array('f', [0.0, 1.0, 0.0])
    norm = normalize_gravity(xyz)
    assert_close_arr(norm, [0.0, 1.0, 0.0])

    # with int16 array
    xyz = array.array('h', [0, 0, 2**15-1])
    norm = normalize_gravity(xyz)
    assert_close_arr(norm, [0.0, 0.0, 1.0])

def test_normalize_gravity_angled():

    xyz = array.array('f', [30.0, 40.0, 0.0])
    # passing an array of where to place output
    out = array.array('f', [0.0, 0.0, 0.0])
    norm = normalize_gravity(xyz, out=out)
    assert norm is out
    assert_close_arr(norm, [0.6, 0.8, 0.0])


def test_normalize_gravity_zero():

    xyz = array.array('f', [0.0, 0.0, 0.0])
    out = array.array('f', [1.0, 1.0, 1.0])  # pre-filled to detect overwrite

    # zero vector should raise Exception
    raised = False
    try:
        normalize_gravity(xyz, out=out)
    except ZeroDivisionError:
        raised = True
    assert raised, "Expected ZeroDivisionError for zero vector"

    # out should not have been touched
    assert_close_arr(out, [1.0, 1.0, 1.0])


def test_normalize_gravity_negative():

    xyz = array.array('f', [-2.0, 0.0, 2.0])
    out = array.array('f', [0.0, 0.0, 0.0])
    norm = normalize_gravity(xyz, out=out)
    expected = [-0.70710677, 0.0, 0.70710677]
    assert norm is out
    assert_close_arr(norm, expected)

def test_normalize_gravity_tiny():
    xyz = array.array('f', [1e-12, 0.0, 0.0])
    out = array.array('f', [0.0, 0.0, 0.0])

    norm = normalize_gravity(xyz, out=out)

    assert_close_arr(norm, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

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

    print("test_orienation.py  (" + str(len(tests)) + " tests)")
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
            raise e

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
