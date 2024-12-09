
# Functions and Modules in Python - Day 3

# Example 1: Defining Functions
def greet(name):
    return f"Hello, {name}!"

# Example usage of greet function
print(greet('Alice'))

# Example 2: Lambda Functions
square_lambda = lambda x: x ** 2
print(square_lambda(4))

# Example 3: Exception Handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print('Divided by zero!')
finally:
    print('This gets executed no matter what.')

# Example 4: Using built-in modules
import math
print(math.sqrt(16))

# Example 5: Creating a simple decorator
def logger(func):
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__} with {args} and {kwargs}')
        return func(*args, **kwargs)
    return wrapper

@logger
def add(x, y):
    return x + y

# Example usage of decorator
print(add(5, 3))

# Example 6: Using datetime module
import datetime
print(datetime.datetime.now())

# Example 7: Creating and using a context manager
class ManagedFile:
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Using the ManagedFile context manager
with ManagedFile('hello.txt') as f:
    f.write('Hello, world!')