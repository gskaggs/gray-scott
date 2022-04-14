
def enum(*sequential, **named):
    # From https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums['from_string'] = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


class RdType:
    VALID_TYPES = set(['gray_scott', 'gierer_mienhardt'])
    def __init__(self, rd_types):
        self.rd_types = set(rd_types)

        self.GRAY_SCOTT = 'gray_scott' in rd_types
        self.GIERER_MIENHARDT = 'gierer_mienhardt' in rd_types

        for type in rd_types:
            assert type in RdType.VALID_TYPES, 'Invalid reaction diffusion type.'
