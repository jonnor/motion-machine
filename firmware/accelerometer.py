
from machine import SoftI2C, Pin
import time
import array
import bma423


def main():

    # how many samples to wait before reading. BMA423 has 1024 bytes FIFO, enough for 150+ samples
    accel_samples = 40
    samplerate = 25

    # pre-allocate buffers
    # raw data (bytes). n_samples X 3 axes X 2 bytes
    accel_buffer = bytearray(accel_samples*3*2)
    # normalized output data (in g). n_samples X 3 axes
    accel_array = array.array('f', list(range(0, accel_samples*3)))

    # setup sensor
    i2c = SoftI2C(scl=11,sda=10)
    sensor = bma423.BMA423(i2c, addr=0x19)
    sensor.fifo_enable()
    sensor.set_accelerometer_freq(samplerate)
    sensor.fifo_clear() # discard any samples lying around in FIFO
    time.sleep_ms(100)

    while True:
        
        # wait until we have enough samples
        fifo_level = sensor.fifo_level()
        if fifo_level >= len(accel_buffer):
            
            # read data
            read_start = time.ticks_ms()
            sensor.fifo_read(accel_buffer)
            read_dur = time.ticks_diff(time.ticks_ms(), read_start)
            
            print('fifo-read', read_start/1000, fifo_level, read_dur)

        
        # limit how often we check
        time.sleep_ms(10)

if __name__ == '__main__':
    main()
