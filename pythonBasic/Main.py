from child import Child

# Create a Child object
kid = Child("ElonMusk", "Money")

# Methods from Parent class
print(kid.introduce())  # Overridden method
print(kid.get_age())    # Inherited method
print(kid.family_rule()) # Inherited method

# Methods from Child class
print(kid.play())
print(kid.child_specific_method())