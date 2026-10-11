"""Send loopback RTP packets over SSH stdout. Python 3.6, no extra dependencies.

The laptop keeps stdin open for the lifetime of its SSH connection. EOF stops this
receiver, so disconnecting releases the UDP port even when the camera is quiet.
"""
import argparse
import socket
import struct
import sys
import threading


def serve(port, output, control):
    stopped = threading.Event()

    def watch_connection():
        try:
            while control.read(1):
                pass
        finally:
            stopped.set()

    watcher = threading.Thread(target=watch_connection)
    watcher.daemon = True
    watcher.start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', port))
        receiver.settimeout(0.5)
        while not stopped.is_set():
            try:
                packet, _ = receiver.recvfrom(65535)
            except socket.timeout:
                continue
            if packet:
                output.write(struct.pack('!H', len(packet)) + packet)
                output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=1234)
    args = parser.parse_args()
    try:
        serve(args.port, sys.stdout.buffer, sys.stdin.buffer)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
