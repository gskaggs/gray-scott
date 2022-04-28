import os

for x in os.listdir():
    with open(x) as f:
        s = f.read()
    s = s.replace('python/sim_driver_tui.py', 'sim_driver_tui.py')
    with open(x, 'w') as f:
        f.write(s)

