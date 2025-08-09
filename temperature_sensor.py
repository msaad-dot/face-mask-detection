from smbus2 import SMBus
from mlx90614 import MLX90614

class TemperatureSensor:
    def __init__(self, address=0x5a):
        bus = SMBus(1)
        self.sensor = MLX90614(bus, address=address)

    def get_temp_celsius(self):
        return self.sensor.get_object_1()

    def get_temp_fahrenheit(self):
        return (self.get_temp_celsius() * 9/5) + 32