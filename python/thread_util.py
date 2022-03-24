from ast import arg
import threading

def run_threads(num_threads, thread_function, args):
    threads = []
    for _ in range(num_threads):
        threads.append(threading.Thread(target=thread_function, args=args))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

class ThreadSafeIterable:
    def __init__(self, data):
        self.data = data
        self.counter = 0
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            # Return None if no more data
            if self.counter >= len(self.data):
                return None

            result = self.data[self.counter]
            self.counter += 1
            return result

    def get_data(self):
        return self.data