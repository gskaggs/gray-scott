from ast import arg
import multiprocessing

def start_processes(num_processes, thread_function, args):
    processes = []
    for _ in range(num_processes):
        processes.append(multiprocessing.Process(target=thread_function, args=args))
    for p in processes:
        p.start()
    return processes

def end_processes(processes):
    for p in processes:
        p.join()

class ProcessSafeIterable:
    def __init__(self, data):
        self.data = data
        self.counter = 0
        self.lock = multiprocessing.Lock()

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