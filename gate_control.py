import RPi.GPIO as GPIO
from time import sleep

class GateControl:
    def __init__(self, green_led=8, red_led=7, servo_pin=15, buzzer=11, ir_sensor=10):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        self.green_led = green_led
        self.red_led = red_led
        self.servo_pin = servo_pin
        self.buzzer = buzzer
        self.ir_sensor = ir_sensor

        GPIO.setup(self.green_led, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.red_led, GPIO.OUT, initial=GPIO.LOW)

        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.servo_pin, 50)
        self.pwm.start(0)

        GPIO.setup(self.buzzer, GPIO.OUT)
        GPIO.output(self.buzzer, GPIO.LOW)

        GPIO.setup(self.ir_sensor, GPIO.IN)

    def open_gate(self):
        self.pwm.ChangeDutyCycle(2.0)
        sleep(0.5)

    def close_gate(self):
        self.pwm.ChangeDutyCycle(12.0)
        sleep(0.1)

    def green_light(self):
        GPIO.output(self.green_led, GPIO.HIGH)
        GPIO.output(self.red_led, GPIO.LOW)

    def red_light(self):
        GPIO.output(self.red_led, GPIO.HIGH)
        GPIO.output(self.green_led, GPIO.LOW)

    def buzz_on(self):
        GPIO.output(self.buzzer, GPIO.HIGH)

    def buzz_off(self):
        GPIO.output(self.buzzer, GPIO.LOW)

    def detect_person(self):
        return GPIO.input(self.ir_sensor) == 0

    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()
