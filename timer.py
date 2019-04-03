import time

def __print_time__(name, diff):
    print("Time used {}: {:.2f}s".format(name, diff))

class SimpleTimer:
    __slots__ = 'time', 'name'

    def __init__(self, name=None):
        self.time = None
        self.name = None
        if name is not None:
            self.set(name)

    def set(self, name):
        self.time = time.time()
        self.name = name

    def report(self, min=0):
        d = time.time() - self.time
        if d > min:
            __print_time__(self.name, d)


class Timer:
    def __init__(self, set_name=None):
        self.timers = {}
        self.last = []
        if set_name is not None:
            self.set(set_name)

    def set(self, name):
        self.timers[name] = time.time()
        self.last.append(name)

    def report(self, name=None):
        if name is None:
            name = self.last.pop()
        if name is None:
            print("Timer: ERROR. Nothing to report.")
        else:
            __print_time__(name, time.time() - self.timers[name])