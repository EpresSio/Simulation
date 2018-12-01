
class Shape:
    def __init__(self, positions, values, Q):
        if len(positions) != len(values):
            raise Exception('Positions and values len should be the same')
        self.Q = Q
        self.positions = positions
        self.values = values
        self.len = len(positions)
