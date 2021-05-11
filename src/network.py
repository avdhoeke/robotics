import socket
import pickle
import struct
import numpy as np


class Network:

    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.host = "192.168.1.40"
        self.port = 9395
        self.addr = (self.host, self.port)
        self.client.connect(self.addr)

    def send(self, data) -> None:
        data = pickle.dumps(data)
        self.client.sendall(struct.pack('>i', len(data)))
        self.client.sendall(data)

    def recv(self) -> list:
        data = self.client.recv(4, socket.MSG_WAITALL)
        data_len = struct.unpack('>i', data)[0]
        data = self.client.recv(data_len, socket.MSG_WAITALL)
        return pickle.loads(data)
