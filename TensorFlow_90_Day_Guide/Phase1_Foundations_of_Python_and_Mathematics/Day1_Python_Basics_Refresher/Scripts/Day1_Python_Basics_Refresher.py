# 🐍 Day 1: Python Basics Refresher 🚀

def try_it_yourself_variables():
    # Define variables
    age = 30
    height = 5.9
    name = "Bob"
    is_employed = True

    # Print variables
    print(age, height, name, is_employed)

def arithmetic_operators_example():
    a = 10
    b = 3
    print(a + b)  # 13
    print(a ** b) # 1000

def comparison_operators_example():
    a = 10
    b = 3
    print(a > b)  # True
    print(a == b) # False

def logical_operators_example():
    a = 10
    b = 3
    print(a > 5 and b < 5)  # True
    print(not(a > 15))      # True

def try_it_yourself_operators():
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

def conditional_statements_example():
    age = 20

    if age >= 18:
        print("You are an adult.")
    elif age > 13:
        print("You are a teenager.")
    else:
        print("You are a child.")

def loops_example():
    # For Loop
    fruits = ["apple", "banana", "cherry"]
    for fruit in fruits:
        print(fruit)

    # While Loop
    count = 0
    while count < 5:
        print(count)
        count += 1

def try_it_yourself_control_structures():
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
    print()  # For newline after the loop

def greet(name):
    return f"Hello, {name}!"

def power(base, exponent=2):
    return base ** exponent

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def try_it_yourself_functions():
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

def importing_module_example():
    import math
    print(math.sqrt(16))  # Output: 4.0

def importing_specific_functions_example():
    from math import pi, sin
    print(pi)          # Output: 3.141592653589793
    print(sin(pi / 2)) # Output: 1.0

def using_own_module_example():
    import math_utils

    print(math_utils.multiply(4, 5))  # Output: 20
    print(math_utils.divide(10, 2))   # Output: 5.0
    print(math_utils.divide(10, 0))   # Output: Cannot divide by zero!

def try_it_yourself_modules():
    import string_utils

    print(string_utils.to_uppercase("hello"))  # Output: HELLO
    print(string_utils.to_lowercase("WORLD"))  # Output: world

def exercise1():
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

def exercise2():
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

def exercise3():
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
    print()  # For newline after the loop

def exercise4():
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

def exercise5():
    import random

    # Generate a random number between 1 and 100
    random_num = random.randint(1, 100)
    print(f"Random Number: {random_num}")

    # Bonus: Using your own module
    import math_utils

    print(math_utils.multiply(4, 5))  # Output: 20
    print(math_utils.divide(10, 2))   # Output: 5.0
    print(math_utils.divide(10, 0))   # Output: Cannot divide by zero!

def interactive_calculator():
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

# Uncomment the functions below to run the respective sections

# try_it_yourself_variables()
# arithmetic_operators_example()
# comparison_operators_example()
# logical_operators_example()
# try_it_yourself_operators()
# conditional_statements_example()
# loops_example()
# try_it_yourself_control_structures()
# greet("Alice")
# print(power(3))
# print(power(3, 3))
# factorial(5)
# try_it_yourself_functions()
# importing_module_example()
# importing_specific_functions_example()
# using_own_module_example()
# try_it_yourself_modules()
# exercise1()
# exercise2()
# exercise3()
# exercise4()
# exercise5()
# interactive_calculator()
