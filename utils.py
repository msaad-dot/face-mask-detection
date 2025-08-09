import subprocess

def send_notification(msg):
    subprocess.Popen(['notify-send', msg])
