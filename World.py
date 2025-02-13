class World:
    def __init__(self):
        self.x = 0
        self.b0 = 0
        self.b1 = 0

    def observe(self):
        return self.x*self.b1 + self.b0

    def act(self):
        pass
    
    def move(self):
        import random
        self.x += random.choice([-1, 1])

