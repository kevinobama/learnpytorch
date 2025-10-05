class Parent:
    def __init__(self, name):
        self.name = name
        self.age = 40

    def introduce(self):
        return f"I am {self.name}, a parent"

    def get_age(self):
        return self.age

    def family_rule(self):
        return "Always be kind to others"