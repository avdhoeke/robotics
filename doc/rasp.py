import numpy as np


class Raspberry:

    def __init__(self, sense):
        self.sense = sense
        self.acceleration_ = None
        self.orientation_ = None
        self.led = [0, 0]

    @property
    def acceleration(self):
        acc = self.sense.get_accelerometer_raw()
        return [acc['x'], acc['y'], acc['z']]

    @property
    def orientation(self):
        gyro = self.sense.get_gyroscope_raw()
        return [gyro['x'], gyro['y'], gyro['z']]

    def place_dot(self, pos: np.ndarray) -> None:
        self.sense.clear()
        pos = pos[0]
        print("Resettting pixel ({}, {})".format(pos[0], pos[1]))
        self.sense.set_pixel(pos[0], pos[1], (255, 0, 0))

    def move_led(self, action: int) -> None:

        self.sense.clear()
        x, y = self.led[0], self.led[1]

        if action == 0:  # Move to the right
            self.led[0] = x-1 if x>0 else x

        if action == 1:  # Move the the left
            self.led[0] = x+1 if x<7 else x

        if action == 2:  # Move upwards
            self.led[1] = y-1 if y>0 else y

        if action == 3:  # Move down
            self.led[1] = y+1 if y<7 else y
        
        if action == 4:  # Stay at the same position
            self.led[0], self.led[1] = x, y
        
        self.sense.set_pixel(self.led[0], self.led[1], (255, 0, 0))
