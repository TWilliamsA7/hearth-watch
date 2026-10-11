"""Bring RTP datagrams from a Jetson SSH session to the local video decoder."""
import os
import socket
import struct
import subprocess
from threading import Event, Lock, Thread


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count - len(result))
        if not chunk:
            raise EOFError('video SSH connection closed')
        result.extend(chunk)
    return bytes(result)


def forward_packets(stream, send):
    while True:
        size, = struct.unpack('!H', read_exact(stream, 2))
        if not 0 < size <= 65507:
            raise ValueError('invalid RTP packet length from SSH')
        send(read_exact(stream, size))


class RtpTunnel:
    def __init__(self, target, port=1234, remote_port=1234, batch=True):
        self.destination = ('127.0.0.1', port)
        self.command = ['ssh.exe' if os.name == 'nt' else 'ssh',
                        '-T', '-o', 'BatchMode=yes' if batch else 'BatchMode=no', '-o', 'ConnectTimeout=5',
                        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
                        target, 'cd ~/hearth-watch/edge && python3 -u -m '
                        'hearth_edge.rtp_tunnel --port %d' % remote_port]
        self.stopped = Event()
        self.lock = Lock()
        self.process = None
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while not self.stopped.is_set():
            process = None
            try:
                with self.lock:
                    if self.stopped.is_set():
                        return
                    process = self.process = subprocess.Popen(
                        self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
                    forward_packets(process.stdout, lambda packet: output.sendto(packet, self.destination))
            except (OSError, EOFError, ValueError) as error:
                if not self.stopped.is_set():
                    print('[relay] video tunnel: %s; reconnecting' % error, flush=True)
            finally:
                if process is not None:
                    # Closing SSH also closes the remote helper's stdin, releasing its socket.
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    process.stdin.close()
                    process.stdout.close()
            self.stopped.wait(2)

    def close(self):
        self.stopped.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.worker.join(timeout=5)
