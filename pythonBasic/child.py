from parent import Parent


class Child(Parent):
    def __init__(self, name, favorite_toy):
        super().__init__(name)  # Call parent constructor
        self.favorite_toy = favorite_toy
        self.age = 10  # Override parent's age

    def play(self):
        return f"{self.name} is playing with {self.favorite_toy}"

    # Override parent method
    def introduce(self):
        return f"I am {self.name}, a child who loves {self.favorite_toy}"

    def child_specific_method(self):
        return "This method only exists in Child class"