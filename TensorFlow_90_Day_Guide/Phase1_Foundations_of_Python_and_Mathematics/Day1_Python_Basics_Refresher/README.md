## 📑 Table of Contents

1. 🌟 Welcome to Day 1
2. 🐍 Python Basics
    - Variables and Data Types
    - Operators
    - Control Structures
    - Functions
    - Modules and Packages
3. 💻 Hands-On Coding
    - Example Script: Interactive Calculator
4. 🧩 Interactive Exercises
5. 📚 Resources
6. 💡 Tips and Tricks

---

## 1. 🌟 Welcome to Day 1

Welcome to **Day 1** of "Becoming a Scikit-Learn Boss in 90 Days"! 🎉 Today, we kickstart our journey by **refreshering Python basics**. Whether you're a seasoned programmer or just starting, revisiting these fundamentals will ensure you're well-prepared for the advanced topics ahead. Let's dive in and set a strong foundation for your machine learning endeavors! 🚀

---

## 2. 🐍 Python Basics

### 📝 Variables and Data Types

**Variables** are the building blocks of any programming language. They store data that can be manipulated and used throughout your code. Python offers various data types to handle different kinds of data.

- **Integers (`int`)**: Whole numbers without a decimal point.
  ```python
  age = 25
  year = 2024
  ```

- **Floating-point (`float`)**: Numbers with decimal points.
  ```python
  pi = 3.1416
  temperature = 36.6
  ```

- **Strings (`str`)**: Sequences of characters enclosed in quotes.
  ```python
  name = "Alice"
  greeting = 'Hello, World!'
  ```

- **Booleans (`bool`)**: Represent logical values, either `True` or `False`.
  ```python
  is_student = True
  has_graduated = False
  ```

#### 🧩 **Try It Yourself!**
Create variables of different data types and print their values.
```python
# Define variables
age = 30
height = 5.9
name = "Bob"
is_employed = True

# Print variables
print(age, height, name, is_employed)
```

---

### ⚙️ Operators

Operators allow you to perform operations on variables and values. Python supports several types of operators:

- **Arithmetic Operators**: `+`, `-`, `*`, `/`, `%`, `**`, `//`
  ```python
  a = 10
  b = 3
  print(a + b)  # 13
  print(a ** b) # 1000
  ```

- **Comparison Operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`
  ```python
  print(a > b)  # True
  print(a == b) # False
  ```

- **Logical Operators**: `and`, `or`, `not`
  ```python
  print(a > 5 and b < 5)  # True
  print(not(a > 15))      # True
  ```

#### 🧩 **Try It Yourself!**
Write a script that takes two numbers as input and performs all arithmetic operations.
```python
# Input numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform operations
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2
division = num1 / num2 if num2 != 0 else "Undefined"

# Display results
print(f"Addition: {addition}")
print(f"Subtraction: {subtraction}")
print(f"Multiplication: {multiplication}")
print(f"Division: {division}")

# Bonus: Comparison
if num1 > num2:
    print(f"{num1} is greater than {num2}.")
elif num1 < num2:
    print(f"{num1} is less than {num2}.")
else:
    print(f"Both numbers are equal.")
```

---

### 🔄 Control Structures

Control structures dictate the flow of your program. The primary ones in Python are **conditional statements** and **loops**.

#### 🔺 Conditional Statements

- **`if` Statement**: Executes a block of code if a condition is true.
- **`elif` Statement**: Checks another condition if the previous `if` was false.
- **`else` Statement**: Executes a block of code if all previous conditions are false.

**Example:**
```python
age = 20

if age >= 18:
    print("You are an adult.")
elif age > 13:
    print("You are a teenager.")
else:
    print("You are a child.")
```

#### 🔁 Loops

- **`for` Loop**: Iterates over a sequence (like a list, tuple, or string).
- **`while` Loop**: Repeats as long as a condition is true.

**Example:**
```python
# For Loop
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# While Loop
count = 0
while count < 5:
    print(count)
    count += 1
```

#### 🧩 **Try It Yourself!**
Create a program that checks if a number is positive, negative, or zero.
```python
# Check if a number is positive, negative, or zero
num = float(input("Enter a number: "))

if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")

# Bonus: Print all even numbers from 1 to 20
print("Even numbers from 1 to 20:")
for i in range(1, 21):
    if i % 2 == 0:
        print(i, end=' ')
```

---

### 🔧 Functions

Functions are reusable blocks of code that perform specific tasks. They help in organizing code and avoiding repetition.

**Defining a Function:**
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

**Functions with Default Parameters:**
```python
def power(base, exponent=2):
    return base ** exponent

print(power(3))       # Uses default exponent 2
print(power(3, 3))    # Uses exponent 3
```

**Example:**
Create a function that calculates the factorial of a number.
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))  # Output: 120
```

#### 🧩 **Try It Yourself!**
Define a function that takes a list of numbers and returns the largest number.
```python
def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

print(find_max([3, 1, 4, 1, 5, 9, 2]))  # Output: 9

# Bonus: Function to check if a string is a palindrome
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print(is_palindrome("Racecar"))  # Output: True
print(is_palindrome("Hello"))    # Output: False
```

---

### 📦 Modules and Packages

**Modules** are Python files containing functions, classes, or variables that you can include in your project. **Packages** are collections of modules organized in directories.

#### 📥 Importing a Module
```python
import math

print(math.sqrt(16))  # Output: 4.0
```

#### 📥 Importing Specific Functions
```python
from math import pi, sin

print(pi)          # Output: 3.141592653589793
print(sin(pi / 2)) # Output: 1.0
```

#### 🛠️ Creating Your Own Module

1. **Create a Python file** (e.g., `math_utils.py`):
    ```python
    # math_utils.py
    def multiply(a, b):
        return a * b

    def divide(a, b):
        if b == 0:
            return "Cannot divide by zero!"
        return a / b
    ```

2. **Import and Use the Module:**
    ```python
    import math_utils

    print(math_utils.multiply(4, 5))  # Output: 20
    print(math_utils.divide(10, 2))   # Output: 5.0
    print(math_utils.divide(10, 0))   # Output: Cannot divide by zero!
    ```

#### 🧩 **Try It Yourself!**
Create your own module with at least two functions and import them into another script.
```python
# Create a file named 'string_utils.py'

def to_uppercase(s):
    return s.upper()

def to_lowercase(s):
    return s.lower()
```

```python
# Another script to import and use 'string_utils.py'

import string_utils

print(string_utils.to_uppercase("hello"))  # Output: HELLO
print(string_utils.to_lowercase("WORLD"))  # Output: world
```

---

## 3. 💻 Hands-On Coding

### 🎉 Example Script: Interactive Calculator 🎉

Let's build a simple interactive calculator that can perform basic arithmetic operations. This script will allow users to choose an operation and input numbers to get results in real-time.

**Script: `Day1_Python_Basics_Refresher.py`**

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error! Division by zero."
    return a / b

def main():
    while True:
        print("\n--- Simple Calculator ---")
        print("Select operation:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Exit")

        choice = input("Enter choice (1/2/3/4/5): ")

        if choice == '5':
            print("Thank you for using the calculator. Goodbye!")
            break

        if choice in ['1', '2', '3', '4']:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
            except ValueError:
                print("Invalid input! Please enter numeric values.")
                continue

            if choice == '1':
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == '2':
                print(f"{num1} - {num2} = {subtract(num1, num2)}")
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == '4':
                result = divide(num1, num2)
                print(f"{num1} / {num2} = {result}")
        else:
            print("Invalid input! Please choose a valid operation.")

if __name__ == "__main__":
    main()
```

#### 🚀 **How to Run:**

1. **Navigate to the `Scripts` directory:**
    ```bash
    cd Phase1_Foundations_of_Python_and_Mathematics/Day1_Python_Basics_Refresher/Scripts
    ```

2. **Execute the script:**
    ```bash
    python Day1_Python_Basics_Refresher.py
    ```

3. **Follow the on-screen instructions to perform calculations.**

#### 🖥️ **Sample Interaction:**

```
--- Simple Calculator ---
Select operation:
1. Add
2. Subtract
3. Multiply
4. Divide
5. Exit
Enter choice (1/2/3/4/5): 1
Enter first number: 10
Enter second number: 5
10.0 + 5.0 = 15.0

--- Simple Calculator ---
Select operation:
1. Add
2. Subtract
3. Multiply
4. Divide
5. Exit
Enter choice (1/2/3/4/5): 4
Enter first number: 10
Enter second number: 0
10.0 / 0.0 = Error! Division by zero.

--- Simple Calculator ---
Select operation:
1. Add
2. Subtract
3. Multiply
4. Divide
5. Exit
Enter choice (1/2/3/4/5): 5
Thank you for using the calculator. Goodbye!
```

---

## 4. 🧩 Interactive Exercises

### 📝 Exercise 1: Variables and Data Types

- **Task**: Declare variables of different data types and print their values.
- **Bonus**: Convert a string to an integer and perform arithmetic operations.

```python
# Define variables
temperature = 23.5    # float
city = "New York"     # string
is_raining = False    # boolean

# Print variables
print(temperature, city, is_raining)

# Bonus: Convert string to integer
str_num = "100"
int_num = int(str_num)
print(int_num + 50)    # Output: 150
```

### 📝 Exercise 2: Operators

- **Task**: Write a script that takes two numbers as input and performs all arithmetic operations.
- **Bonus**: Compare two numbers and print which one is greater or if they are equal.

```python
# Input numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform operations
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2
division = num1 / num2 if num2 != 0 else "Undefined"

# Display results
print(f"Addition: {addition}")
print(f"Subtraction: {subtraction}")
print(f"Multiplication: {multiplication}")
print(f"Division: {division}")

# Bonus: Comparison
if num1 > num2:
    print(f"{num1} is greater than {num2}.")
elif num1 < num2:
    print(f"{num1} is less than {num2}.")
else:
    print(f"Both numbers are equal.")
```

### 📝 Exercise 3: Control Structures

- **Task**: Create a program that checks if a number is positive, negative, or zero.
- **Bonus**: Write a loop that prints all even numbers from 1 to 20.

```python
# Check if a number is positive, negative, or zero
num = float(input("Enter a number: "))

if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")

# Bonus: Print all even numbers from 1 to 20
print("Even numbers from 1 to 20:")
for i in range(1, 21):
    if i % 2 == 0:
        print(i, end=' ')
```

### 📝 Exercise 4: Functions

- **Task**: Define a function that takes a list of numbers and returns the largest number.
- **Bonus**: Create a function that checks if a string is a palindrome.

```python
# Function to find the largest number in a list
def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

print(find_max([3, 1, 4, 1, 5, 9, 2]))  # Output: 9

# Bonus: Function to check if a string is a palindrome
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print(is_palindrome("Racecar"))  # Output: True
print(is_palindrome("Hello"))    # Output: False
```

### 📝 Exercise 5: Modules and Packages

- **Task**: Use the `random` module to generate a random number between 1 and 100.
- **Bonus**: Create your own module with at least two functions and import them into another script.

```python
import random

# Generate a random number between 1 and 100
random_num = random.randint(1, 100)
print(f"Random Number: {random_num}")

# Bonus: Using your own module
import math_utils

print(math_utils.multiply(4, 5))  # Output: 20
print(math_utils.divide(10, 2))   # Output: 5.0
print(math_utils.divide(10, 0))   # Output: Cannot divide by zero!
```

---

## 5. 📚 Resources

Enhance your learning with these excellent resources:

- [**Official Python Documentation**](https://docs.python.org/3/)
- [**W3Schools Python Tutorial**](https://www.w3schools.com/python/)
- [**Real Python**](https://realpython.com/)
- [**Python for Everybody (Coursera)**](https://www.coursera.org/specializations/python)
- [**Automate the Boring Stuff with Python**](https://automatetheboringstuff.com/)
- [**Codecademy Python Course**](https://www.codecademy.com/learn/learn-python-3)
- [**LeetCode Python Problems**](https://leetcode.com/problemset/all/?difficulty=Easy&listId=wpwgkgt)

---

## 6. 💡 Tips and Tricks

### 💡 Pro Tip

**Virtual Environments**: Always use virtual environments to manage your project dependencies. This keeps your projects isolated and prevents version conflicts.

```bash
# Create a virtual environment
python3 -m venv my_env

# Activate the virtual environment
source my_env/bin/activate  # On Windows: my_env\Scripts\activate

# Install packages
pip install package_name
```

### 🛠️ Recommended Tools

- **Visual Studio Code**: A powerful code editor with Python extensions.
- **PyCharm**: An IDE specifically designed for Python development.
- **Jupyter Notebook**: Interactive notebooks for data analysis and visualization.

### 🚀 Speed Up Your Coding

- **Use List Comprehensions**: They provide a concise way to create lists.
  ```python
  squares = [x**2 for x in range(10)]
  ```
- **Leverage Built-in Functions**: Python's standard library offers a plethora of useful functions.
  ```python
  numbers = [1, 2, 3, 4, 5]
  print(sum(numbers))  # Output: 15
  print(max(numbers))  # Output: 5
  ```

### 🔍 Debugging Tips

- **Use Print Statements**: Simple yet effective for tracking variable values.
- **Leverage Debuggers**: Tools like the built-in debugger in VS Code can help step through your code.
- **Handle Exceptions**: Gracefully handle errors to prevent your program from crashing.
  ```python
  try:
      result = 10 / 0
  except ZeroDivisionError:
      print("Cannot divide by zero!")
  ```

---


   ![Thank You Animation](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)
    