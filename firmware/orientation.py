
import math
import array

import emlearn_iir

#print(dir(emlearn_iir))
    
#emlearn_iir.new([0.0])

class GravityEstimatorLowpass:
    """Estimate gravity vector from IMU data using a low-pass"""
    
    def __init__(self, coefficients : array.array):
        """
        """
        self.gravity = array.array('f', [0.0, 0.0, 0.0])

        # one filter per XYZ axis
        self.filters = [ emlearn_iir.new(coefficients) for i in range(3) ]


    def update(self, accel : array.array):
        """
        accel: Accelerometer raw data [ax, ay, az]

        Returns: estimated gravity vector - in same unit as input
        """

        if len(accel) % 3 != 0:
            raise ValueError("Input must have 3 columns")

        # Buffer for deinterleaved data
        n_samples = len(accel) // 3

        # Deinterleave and run IIR filter sample-by-sample
        # NOTE: emlearn_iir can take an array of samples as input,
        # but when we would have to deinterleave temporary array anyway
        arr = array.array('f', [0.0])
        for axis in range(0, 3):
            filter_func = self.filters[axis].run
            for sample in range(n_samples):
                index = (sample*3)+axis
                arr[0] = accel[index]
                out = filter_func(arr)
                self.gravity[axis] = arr[0]

        return self.gravity


def normalize_gravity(v, out=None):
    """Compute norimalized gravity vector - aka orientation vector"""

    gx, gy, gz = v
    mag = math.sqrt(gx**2 + gy**2 + gz**2)
    if out is None:
        out = [0.0, 0.0, 0.0]

    out[0] = gx/mag
    out[1] = gy/mag
    out[2] = gz/mag
    return out

def compute_tilt(orientation):
    """
    Compute pitch and roll from orientation vector - in degrees
    Pitch: -90° to +90°
    Roll: -180° to +180°
    NOTE: input vector must be normalized
    NOTE: roll is poorly defined when pitch is near 90 degrees
    """
    gx, gy, gz = orientation

    pitch = math.degrees(math.atan2(gx, math.sqrt(gy**2 + gz**2)))
    roll = math.degrees(math.atan2(gy, gz))

    return pitch, roll

