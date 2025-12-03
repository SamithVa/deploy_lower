import struct


class KeyMap:
    R1 = 4
    L1 = 5
    start = 9
    select = 8
    R2 = 6
    L2 = 7
    A = 2
    B = 1
    X = 3
    Y = 0


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 10

    def set(self, data):
        # wireless_remote
        #keys = struct.unpack("H", data[2:4])[0]
        for i in range(10):
            self.button[i] = (data.keys & (1 << i)) >> i
        self.lx = data.lx
        self.rx = data.rx
        self.ry = data.ry
        self.ly = data.ly
