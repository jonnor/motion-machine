
"""
Generate a low-pass filter for
"""

import json
import argparse
import math

import numpy
from scipy.signal import iirfilter

def create_lowpass(samplerate, cutoff=0.5, order=4):
    nyquist = samplerate / 2
    normalized_cutoff = cutoff / nyquist
    
    # NOTE: in theory an Elliptic filter would allow sharper transition
    sos = iirfilter(order, normalized_cutoff, 
                   btype='lowpass', ftype='butter', output='sos')

    return sos


def parse():
    parser = argparse.ArgumentParser(description='Generate low-pass filter coefficients')

    parser.add_argument('--samplerate', default=25.0, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float)
    parser.add_argument('--order', default=4, type=int)
    parser.add_argument('--out',   default='firmware/orientation_lowpass.json')

    args = parser.parse_args()
    return args


def main():
    args = parse()

    sos = create_lowpass(args.samplerate, cutoff=args.cutoff, order=args.order)
    coefficients = list(sos.flatten())
    
    n_stages = math.ceil(args.order/2)
    assert len(coefficients) == n_stages*6, (len(coefficients), n_stages*6, sos.shape)

    # include metadata in output
    out = {
        'samplerate': args.samplerate,
        'coefficients': coefficients,
        'order': args.order,
        'cutoff': args.cutoff,
    }

    # store as JSON - common
    out_path = args.out
    with open(out_path, 'w') as f:
        json.dump(out, f)
        print('Wrote to', out_path)

    # check it
    with open(out_path, 'r') as f:
        loaded = json.load(f)
        print(loaded)
 
if __name__ == '__main__':
    main()    

