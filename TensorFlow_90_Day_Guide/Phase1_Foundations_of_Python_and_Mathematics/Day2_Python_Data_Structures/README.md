# 🐍 Day 2: Python Data Structures 🛠️

![Python Data Structures](https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif)

---

## 📑 Table of Contents

1. 🌟 Welcome to Day 2
2. 📦 Python Data Structures
    - **Lists**
    - **Tuples**
    - **Dictionaries**
    - **Sets**
    - **Comprehensions**
    - **Nested Data Structures**
3. 💻 Hands-On Coding
    - Example Scripts
4. 🧩 Interactive Exercises
5. 📚 Resources
6. 💡 Tips and Tricks

---

## 1. 🌟 Welcome to Day 2

Welcome to **Day 2** of "Becoming a Scikit-Learn Boss in 90 Days"! 🎉 Today, we delve into **Python Data Structures**, the backbone of efficient programming and data manipulation. Understanding these structures is crucial for handling data effectively, especially in machine learning tasks. Let's explore the powerful tools Python offers to organize and manage data seamlessly! 🚀

---

## 2. 📦 Python Data Structures

Python provides a variety of data structures to store and manipulate data efficiently. Each structure has its unique characteristics and use-cases.

### 📝 Lists

**Lists** are ordered, mutable collections that can hold heterogeneous items. They are defined using square brackets `[]`.

- **Creation and Basic Operations:**

  ```python
  # Creating a list
  fruits = ["apple", "banana", "cherry"]
  
  # Accessing elements
  print(fruits[0])  # Output: apple
  
  # Modifying elements
  fruits[1] = "blueberry"
  print(fruits)  # Output: ['apple', 'blueberry', 'cherry']
  
  # Adding elements
  fruits.append("date")
  print(fruits)  # Output: ['apple', 'blueberry', 'cherry', 'date']
  
  # Removing elements
  fruits.remove("blueberry")
  print(fruits)  # Output: ['apple', 'cherry', 'date']
  ```

- **List Slicing:**

  ```python
  numbers = [0, 1, 2, 3, 4, 5, 6]
  print(numbers[2:5])  # Output: [2, 3, 4]
  print(numbers[:3])   # Output: [0, 1, 2]
  print(numbers[4:])   # Output: [4, 5, 6]
  ```

- **List Methods:**

  ```python
  numbers = [1, 2, 3, 4, 5]
  
  # Insert
  numbers.insert(2, 99)
  print(numbers)  # Output: [1, 2, 99, 3, 4, 5]
  
  # Pop
  popped = numbers.pop()
  print(popped)   # Output: 5
  print(numbers)  # Output: [1, 2, 99, 3, 4]
  
  # Sort
  numbers.sort()
  print(numbers)  # Output: [1, 2, 3, 4, 99]
  
  # Reverse
  numbers.reverse()
  print(numbers)  # Output: [99, 4, 3, 2, 1]
  ```

#### 🧩 **Try It Yourself!**
Create a list of your favorite movies, add a new movie, remove one, and print the updated list.
```python
# Define the list
favorite_movies = ["Inception", "The Matrix", "Interstellar"]

# Add a new movie
favorite_movies.append("The Prestige")
print(favorite_movies)  # Output: ['Inception', 'The Matrix', 'Interstellar', 'The Prestige']

# Remove a movie
favorite_movies.remove("The Matrix")
print(favorite_movies)  # Output: ['Inception', 'Interstellar', 'The Prestige']
```

---

### 📝 Tuples

**Tuples** are ordered, immutable collections that can hold heterogeneous items. They are defined using parentheses `()`.

- **Creation and Basic Operations:**

  ```python
  # Creating a tuple
  coordinates = (10.0, 20.0, 30.0)
  
  # Accessing elements
  print(coordinates[1])  # Output: 20.0
  
  # Tuples are immutable
  # coordinates[1] = 25.0  # This will raise a TypeError
  ```

- **Tuple Packing and Unpacking:**

  ```python
  # Packing
  person = ("Alice", 30, "Engineer")
  
  # Unpacking
  name, age, profession = person
  print(name)        # Output: Alice
  print(age)         # Output: 30
  print(profession)  # Output: Engineer
  ```

- **Tuple Methods:**

  ```python
  numbers = (1, 2, 3, 2, 4, 2, 5)
  
  # Count occurrences
  count_twos = numbers.count(2)
  print(count_twos)  # Output: 3
  
  # Find index of first occurrence
  index_three = numbers.index(3)
  print(index_three)  # Output: 2
  ```

#### 🧩 **Try It Yourself!**
Create a tuple with your personal information and unpack it.
```python
# Define the tuple
personal_info = ("John Doe", 28, "Data Scientist")

# Unpack the tuple
name, age, occupation = personal_info
print(name)        # Output: John Doe
print(age)         # Output: 28
print(occupation)  # Output: Data Scientist
```

---

### 📝 Dictionaries

**Dictionaries** are unordered, mutable collections of key-value pairs. They are defined using curly braces `{}`.

- **Creation and Basic Operations:**

  ```python
  # Creating a dictionary
  student = {
      "name": "Emily",
      "age": 22,
      "major": "Computer Science"
  }
  
  # Accessing values
  print(student["name"])  # Output: Emily
  
  # Modifying values
  student["age"] = 23
  print(student)  # Output: {'name': 'Emily', 'age': 23, 'major': 'Computer Science'}
  
  # Adding new key-value pair
  student["graduated"] = False
  print(student)  # Output: {'name': 'Emily', 'age': 23, 'major': 'Computer Science', 'graduated': False}
  
  # Removing a key-value pair
  del student["graduated"]
  print(student)  # Output: {'name': 'Emily', 'age': 23, 'major': 'Computer Science'}
  ```

- **Dictionary Methods:**

  ```python
  student = {
      "name": "Emily",
      "age": 23,
      "major": "Computer Science"
  }
  
  # Get keys
  keys = student.keys()
  print(keys)  # Output: dict_keys(['name', 'age', 'major'])
  
  # Get values
  values = student.values()
  print(values)  # Output: dict_values(['Emily', 23, 'Computer Science'])
  
  # Get items
  items = student.items()
  print(items)  # Output: dict_items([('name', 'Emily'), ('age', 23), ('major', 'Computer Science')])
  
  # Get value with default
  grade = student.get("grade", "A")
  print(grade)  # Output: A
  ```

#### 🧩 **Try It Yourself!**
Create a dictionary to store information about a book, update its price, and print the updated dictionary.
```python
# Define the dictionary
book = {
    "title": "Python Programming",
    "author": "Jane Smith",
    "price": 29.99
}

# Update the price
book["price"] = 24.99
print(book)  # Output: {'title': 'Python Programming', 'author': 'Jane Smith', 'price': 24.99}

# Add a new key-value pair
book["in_stock"] = True
print(book)  # Output: {'title': 'Python Programming', 'author': 'Jane Smith', 'price': 24.99, 'in_stock': True}
```

---

### 📝 Sets

**Sets** are unordered, mutable collections of unique items. They are defined using curly braces `{}` or the `set()` function.

- **Creation and Basic Operations:**

  ```python
  # Creating a set
  colors = {"red", "green", "blue"}
  
  # Adding elements
  colors.add("yellow")
  print(colors)  # Output: {'red', 'green', 'blue', 'yellow'}
  
  # Removing elements
  colors.remove("green")
  print(colors)  # Output: {'red', 'blue', 'yellow'}
  
  # Sets automatically eliminate duplicates
  colors.add("red")
  print(colors)  # Output: {'red', 'blue', 'yellow'}
  ```

- **Set Operations:**

  ```python
  set1 = {1, 2, 3, 4}
  set2 = {3, 4, 5, 6}
  
  # Union
  union_set = set1.union(set2)
  print(union_set)  # Output: {1, 2, 3, 4, 5, 6}
  
  # Intersection
  intersection_set = set1.intersection(set2)
  print(intersection_set)  # Output: {3, 4}
  
  # Difference
  difference_set = set1.difference(set2)
  print(difference_set)  # Output: {1, 2}
  
  # Symmetric Difference
  sym_diff_set = set1.symmetric_difference(set2)
  print(sym_diff_set)  # Output: {1, 2, 5, 6}
  ```

#### 🧩 **Try It Yourself!**
Create a set of unique programming languages, add a new language, and perform a union with another set.
```python
# Define the set
languages = {"Python", "Java", "C++"}

# Add a new language
languages.add("JavaScript")
print(languages)  # Output: {'Python', 'Java', 'C++', 'JavaScript'}

# Define another set
new_languages = {"Ruby", "Go", "Python"}

# Perform union
all_languages = languages.union(new_languages)
print(all_languages)  # Output: {'Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'Go'}
```

---

### 📝 Comprehensions

**Comprehensions** provide a concise way to create lists, dictionaries, and sets.

- **List Comprehensions:**

  ```python
  # Create a list of squares
  squares = [x**2 for x in range(10)]
  print(squares)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
  
  # With condition
  even_squares = [x**2 for x in range(10) if x % 2 == 0]
  print(even_squares)  # Output: [0, 4, 16, 36, 64]
  ```

- **Dictionary Comprehensions:**

  ```python
  # Create a dictionary of squares
  squares_dict = {x: x**2 for x in range(5)}
  print(squares_dict)  # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
  
  # With condition
  squares_dict = {x: x**2 for x in range(5) if x % 2 == 0}
  print(squares_dict)  # Output: {0: 0, 2: 4, 4: 16}
  ```

- **Set Comprehensions:**

  ```python
  # Create a set of squares
  squares_set = {x**2 for x in range(5)}
  print(squares_set)  # Output: {0, 1, 4, 9, 16}
  
  # With condition
  squares_set = {x**2 for x in range(5) if x % 2 != 0}
  print(squares_set)  # Output: {1, 9}
  ```

#### 🧩 **Try It Yourself!**
Use list comprehensions to create a list of even numbers and a dictionary mapping numbers to their cubes.
```python
# List of even numbers from 1 to 20
even_numbers = [x for x in range(1, 21) if x % 2 == 0]
print(even_numbers)  # Output: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Dictionary mapping numbers to their cubes
cubes = {x: x**3 for x in range(1, 6)}
print(cubes)  # Output: {1: 1, 2: 8, 3: 27, 4: 64, 5: 125}
```

---

### 📝 Nested Data Structures

Python allows nesting of data structures, enabling the creation of complex and hierarchical data.

- **Nested Lists:**

  ```python
  matrix = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
  ]
  
  # Accessing elements
  print(matrix[0][1])  # Output: 2
  ```

- **Nested Dictionaries:**

  ```python
  students = {
      "Alice": {"age": 25, "major": "Physics"},
      "Bob": {"age": 22, "major": "Mathematics"},
      "Charlie": {"age": 23, "major": "Computer Science"}
  }
  
  # Accessing nested values
  print(students["Bob"]["major"])  # Output: Mathematics
  ```

- **Mixed Nested Structures:**

  ```python
  data = {
      "fruits": ["apple", "banana", "cherry"],
      "vegetables": {"root": "carrot", "leafy": "spinach"},
      "grains": {"cereals": ["rice", "wheat"], "pseudocereals": ["quinoa", "buckwheat"]}
  }
  
  # Accessing nested elements
  print(data["grains"]["cereals"][1])  # Output: wheat
  print(data["vegetables"]["leafy"])   # Output: spinach
  ```

#### 🧩 **Try It Yourself!**
Create a nested dictionary representing a company's departments and employees.
```python
# Define the nested dictionary
company = {
    "Engineering": {
        "Alice": {"age": 30, "role": "Software Engineer"},
        "Bob": {"age": 25, "role": "DevOps Engineer"}
    },
    "HR": {
        "Charlie": {"age": 28, "role": "HR Manager"},
        "Diana": {"age": 26, "role": "Recruiter"}
    }
}

# Accessing nested information
print(company["Engineering"]["Alice"]["role"])  # Output: Software Engineer
print(company["HR"]["Diana"]["age"])           # Output: 26
```

---

## 3. 💻 Hands-On Coding

### 🎉 Example Scripts

#### 📝 Script 1: Managing a List of Students

```python
# Define a list of students
students = ["Alice", "Bob", "Charlie"]

# Add a new student
students.append("Diana")
print(students)  # Output: ['Alice', 'Bob', 'Charlie', 'Diana']

# Remove a student
students.remove("Bob")
print(students)  # Output: ['Alice', 'Charlie', 'Diana']

# Iterate through the list
for student in students:
    print(f"Student: {student}")
```

#### 📝 Script 2: Working with Dictionaries

```python
# Define a dictionary of students
students = {
    "Alice": {"age": 25, "major": "Physics"},
    "Charlie": {"age": 23, "major": "Computer Science"},
    "Diana": {"age": 26, "major": "Mathematics"}
}

# Add a new student
students["Eve"] = {"age": 22, "major": "Biology"}
print(students)

# Update a student's major
students["Alice"]["major"] = "Astronomy"
print(students["Alice"])

# Iterate through the dictionary
for name, info in students.items():
    print(f"{name} is {info['age']} years old and majors in {info['major']}.")
```

#### 📝 Script 3: Set Operations

```python
# Define two sets
set1 = {"apple", "banana", "cherry"}
set2 = {"banana", "dragonfruit", "elderberry"}

# Union
union_set = set1.union(set2)
print(f"Union: {union_set}")  # Output: {'apple', 'banana', 'cherry', 'dragonfruit', 'elderberry'}

# Intersection
intersection_set = set1.intersection(set2)
print(f"Intersection: {intersection_set}")  # Output: {'banana'}

# Difference
difference_set = set1.difference(set2)
print(f"Difference: {difference_set}")  # Output: {'apple', 'cherry'}

# Symmetric Difference
sym_diff_set = set1.symmetric_difference(set2)
print(f"Symmetric Difference: {sym_diff_set}")  # Output: {'apple', 'cherry', 'dragonfruit', 'elderberry'}
```

#### 📝 Script 4: Using Comprehensions

```python
# List comprehension to create a list of squares
squares = [x**2 for x in range(10)]
print(squares)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Dictionary comprehension to map numbers to their cubes
cubes = {x: x**3 for x in range(5)}
print(cubes)  # Output: {0: 0, 1: 1, 2: 8, 3: 27, 4: 64}

# Set comprehension to create a set of unique vowels in a word
word = "banana"
vowels = {letter for letter in word if letter in 'aeiou'}
print(vowels)  # Output: {'a'}
```

---

## 4. 🧩 Interactive Exercises

### 📝 Exercise 1: Lists

- **Task**: Create a list of your favorite books, add a new book, remove one, and print the updated list.
  
  ```python
  # Define the list
  favorite_books = ["1984", "To Kill a Mockingbird", "The Great Gatsby"]
  
  # Add a new book
  favorite_books.append("Brave New World")
  print(favorite_books)  # Output: ['1984', 'To Kill a Mockingbird', 'The Great Gatsby', 'Brave New World']
  
  # Remove a book
  favorite_books.remove("1984")
  print(favorite_books)  # Output: ['To Kill a Mockingbird', 'The Great Gatsby', 'Brave New World']
  ```

### 📝 Exercise 2: Tuples

- **Task**: Create a tuple with your favorite cities and attempt to modify one element (observe the result).
  
  ```python
  # Define the tuple
  favorite_cities = ("New York", "Paris", "Tokyo")
  
  # Attempt to modify an element
  try:
      favorite_cities[1] = "London"
  except TypeError as e:
      print(e)  # Output: 'tuple' object does not support item assignment
  ```

### 📝 Exercise 3: Dictionaries

- **Task**: Create a dictionary to store information about a car, update its mileage, and print the updated dictionary.
  
  ```python
  # Define the dictionary
  car = {
      "make": "Toyota",
      "model": "Camry",
      "year": 2018,
      "mileage": 50000
  }
  
  # Update mileage
  car["mileage"] += 1500
  print(car)  # Output: {'make': 'Toyota', 'model': 'Camry', 'year': 2018, 'mileage': 51500}
  
  # Add a new key-value pair
  car["color"] = "Blue"
  print(car)  # Output: {'make': 'Toyota', 'model': 'Camry', 'year': 2018, 'mileage': 51500, 'color': 'Blue'}
  ```

### 📝 Exercise 4: Sets

- **Task**: Create two sets of your favorite sports, perform union and intersection operations, and print the results.
  
  ```python
  # Define two sets
  favorite_sports1 = {"soccer", "basketball", "tennis"}
  favorite_sports2 = {"tennis", "swimming", "cricket"}
  
  # Union
  union_sports = favorite_sports1.union(favorite_sports2)
  print(f"Union: {union_sports}")  # Output: {'soccer', 'basketball', 'tennis', 'swimming', 'cricket'}
  
  # Intersection
  intersection_sports = favorite_sports1.intersection(favorite_sports2)
  print(f"Intersection: {intersection_sports}")  # Output: {'tennis'}
  ```

### 📝 Exercise 5: Comprehensions

- **Task**: Use a list comprehension to create a list of squares for even numbers between 1 and 10.
  
  ```python
  # List comprehension for squares of even numbers
  even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
  print(even_squares)  # Output: [4, 16, 36, 64, 100]
  ```

### 📝 Exercise 6: Nested Data Structures

- **Task**: Create a nested dictionary representing a library with books categorized by genre.
  
  ```python
  # Define the nested dictionary
  library = {
      "Fiction": {
          "1984": {"author": "George Orwell", "copies": 4},
          "The Great Gatsby": {"author": "F. Scott Fitzgerald", "copies": 2}
      },
      "Non-Fiction": {
          "Sapiens": {"author": "Yuval Noah Harari", "copies": 5},
          "Educated": {"author": "Tara Westover", "copies": 3}
      }
  }
  
  # Accessing nested information
  print(library["Fiction"]["1984"]["author"])  # Output: George Orwell
  
  # Adding a new book
  library["Fiction"]["Brave New World"] = {"author": "Aldous Huxley", "copies": 3}
  print(library["Fiction"])
  # Output: {'1984': {'author': 'George Orwell', 'copies': 4}, 'The Great Gatsby': {'author': 'F. Scott Fitzgerald', 'copies': 2}, 'Brave New World': {'author': 'Aldous Huxley', 'copies': 3}}
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
