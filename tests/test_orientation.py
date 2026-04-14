
import sys
import array

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

# -----
# Tests
# -----
def test_gravity_estimator():

    pass
    #est = GravityEstimatorLowpass()

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
