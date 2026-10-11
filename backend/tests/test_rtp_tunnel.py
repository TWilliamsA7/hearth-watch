import io
from pathlib import Path
from queue import Empty, Queue
import socket
import struct
import subprocess
import sys
from threading import Thread
import time

import pytest

from hearth.perception.rtp_tunnel import forward_packets


class Fragmented(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


def test_packet_boundaries_survive_fragmented_ssh_reads():
    packets = [b'\x80\x60\x00\x01payload', bytes(range(256)) * 5]
    stream = Fragmented(b''.join(struct.pack('!H', len(p)) + p for p in packets))
    received = []
    with pytest.raises(EOFError):
        forward_packets(stream, received.append)
    assert received == packets


@pytest.mark.parametrize('wire', [b'\x00', b'\x00\x05ab'])
def test_truncated_packet_is_never_sent_to_decoder(wire):
    received = []
    with pytest.raises(EOFError):
        forward_packets(io.BytesIO(wire), received.append)
    assert received == []


@pytest.mark.parametrize('size', [0, 65508, 65535])
def test_invalid_packet_length_is_rejected(size):
    with pytest.raises(ValueError):
        forward_packets(io.BytesIO(struct.pack('!H', size)), lambda p: pytest.fail('sent invalid packet'))


def test_remote_helper_forwards_udp_and_releases_port_on_ssh_eof():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    helper = Path(__file__).resolve().parents[2] / 'edge/hearth_edge/rtp_tunnel.py'
    process = subprocess.Popen([sys.executable, '-u', str(helper), '--port', str(port)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    received = Queue()

    def read():
        try:
            forward_packets(process.stdout, received.put)
        except EOFError:
            pass

    reader = Thread(target=read, daemon=True)
    reader.start()
    packet = b'\x80\x60\x00\x01\x00\xff\r\nRTP'
    try:
        deadline = time.monotonic() + 5
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            while True:
                sender.sendto(packet, ('127.0.0.1', port))
                try:
                    assert received.get(timeout=.1) == packet
                    break
                except Empty:
                    assert process.poll() is None, process.stderr.read().decode()
                    assert time.monotonic() < deadline, 'helper did not forward RTP'
        # EOF must free the port even with no more video packets arriving.
        process.stdin.close()
        assert process.wait(timeout=3) == 0
        reader.join(timeout=2)
        assert not reader.is_alive()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(('127.0.0.1', port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
