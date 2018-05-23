import random
import matplotlib.pyplot as plt

class Color:
    def __init__(self, n=20):
        self.cm = plt.get_cmap('hsv')
        self.i = 0
        self.n = n

    @staticmethod
    def get_rand_color():
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def get_next_color(self):
        c = self.cm(1.*self.i/self.n)
        c = [int(round(ci*255)) for ci in c]
        self.i = (self.i + self.n/3) % self.n
        return c[0:3]
