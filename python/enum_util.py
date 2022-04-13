# From https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums['from_string'] = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
