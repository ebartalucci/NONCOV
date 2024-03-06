import socket
import time

# Address and port of the Processing sketch
HOST = '127.0.0.1'
PORT = 5204

# Function to send a command to the Processing sketch
def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())

# Send a command to start audio processing in the Processing sketch
send_command("startAudio")

# Do other tasks in your Python script
# For example:
time.sleep(10)  # Simulate some other processing
print("Finished processing")
