from . import Network


class Agent:

    def __init__(self):
        self.network = Network()

    def run(self):
        while True:
            self.network.send("Hello!")
            data = self.network.recv()
            print(data)