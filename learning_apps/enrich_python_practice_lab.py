"""
Enrich Python Practice Lab curriculum from 8 → 28+ items.
Covers Python fundamentals, data structures, algorithms, NumPy/Pandas, functional programming,
OOP, testing, decorators, generators, context managers, and problem-solving patterns.
"""
import json
from pathlib import Path

def generate_curriculum_items():
    """Generate comprehensive Python practice curriculum."""
    
    items = []
    
    # ============================================================================
    # PYTHON FUNDAMENTALS (7 items)
    # ============================================================================
    
    items.append({
        "id": "py_basics",
        "book_id": "fundamentals",
        "level": "basics",
        "title": "Python Basics: Types, Variables, Control Flow",
        "learn": "Variables, types (int, float, str, bool), operators, if/elif/else, for/while loops, range(), break/continue. Foundation for all Python programming.",
        "try_code": "# Python basics\nx = 42  # int\ny = 3.14  # float\nname = 'Python'  # str\nis_cool = True  # bool\n\n# Control flow\nif x > 40:\n    print('Large')\nelif x > 20:\n    print('Medium')\nelse:\n    print('Small')\n\n# Loops\nfor i in range(5):\n    print(i, end=' ')\nprint()\n\ncount = 0\nwhile count < 3:\n    count += 1\n    if count == 2:\n        continue\n    print(count)",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "py_functions",
        "book_id": "fundamentals",
        "level": "basics",
        "title": "Functions: Definition, Arguments, Return",
        "learn": "def, parameters, arguments (positional, keyword, default, *args, **kwargs), return values, docstrings. Functions are first-class objects in Python.",
        "try_code": "# Function basics\ndef greet(name, greeting='Hello'):\n    \"\"\"Greet someone with optional custom greeting.\"\"\"\n    return f'{greeting}, {name}!'\n\nprint(greet('Alice'))\nprint(greet('Bob', greeting='Hi'))\n\n# Variable arguments\ndef sum_all(*args):\n    return sum(args)\n\nprint(sum_all(1, 2, 3, 4, 5))\n\n# Keyword arguments\ndef configure(**kwargs):\n    for key, value in kwargs.items():\n        print(f'{key} = {value}')\n\nconfigure(learning_rate=0.01, epochs=100, batch_size=32)",
        "try_demo": None,
        "prerequisites": ["py_basics"]
    })
    
    items.append({
        "id": "py_collections",
        "book_id": "fundamentals",
        "level": "basics",
        "title": "Collections: Lists, Tuples, Dicts, Sets",
        "learn": "list (mutable sequence), tuple (immutable sequence), dict (key-value), set (unique elements). Comprehensions, slicing, common operations.",
        "try_code": "# Lists: mutable, ordered\nnums = [1, 2, 3, 4, 5]\nnums.append(6)\nprint(nums[1:4])  # slicing\nsquares = [x**2 for x in nums]  # list comprehension\n\n# Tuples: immutable, ordered\npoint = (10, 20)\nx, y = point  # unpacking\n\n# Dicts: key-value mapping\nscores = {'Alice': 95, 'Bob': 87, 'Charlie': 92}\nscores['David'] = 88\nfor name, score in scores.items():\n    print(f'{name}: {score}')\n\n# Sets: unique elements\nunique = {1, 2, 2, 3, 3, 3}\nprint(unique)  # {1, 2, 3}\n\n# Set operations\nset1 = {1, 2, 3}\nset2 = {3, 4, 5}\nprint(set1 & set2)  # intersection\nprint(set1 | set2)  # union",
        "try_demo": None,
        "prerequisites": ["py_basics"]
    })
    
    items.append({
        "id": "py_strings",
        "book_id": "fundamentals",
        "level": "basics",
        "title": "String Operations and Formatting",
        "learn": "String methods (split, join, strip, replace, find), slicing, f-strings, format(), str.format(). Strings are immutable sequences.",
        "try_code": "# String operations\ntext = '  Hello, World!  '\nprint(text.strip())  # remove whitespace\nprint(text.lower())\nprint(text.replace('World', 'Python'))\n\nwords = text.strip().split(', ')\nprint(words)\njoined = ' | '.join(words)\nprint(joined)\n\n# String formatting\nname = 'Alice'\nage = 30\n\n# f-strings (Python 3.6+)\nprint(f'{name} is {age} years old')\nprint(f'In 5 years: {age + 5}')\n\n# format() method\nprint('{} scored {:.2f}%'.format(name, 95.678))\n\n# String slicing\nalphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'\nprint(alphabet[::2])  # every 2nd char\nprint(alphabet[::-1])  # reverse",
        "try_demo": None,
        "prerequisites": ["py_basics"]
    })
    
    items.append({
        "id": "py_comprehensions",
        "book_id": "fundamentals",
        "level": "intermediate",
        "title": "Comprehensions: List, Dict, Set, Generator",
        "learn": "Pythonic way to create collections: [expr for x in iterable if cond]. More readable than explicit loops. Generator expressions for memory efficiency.",
        "try_code": "# List comprehension\nsquares = [x**2 for x in range(10)]\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n\n# Dict comprehension\nsquare_dict = {x: x**2 for x in range(5)}\nprint(square_dict)\n\n# Set comprehension\nremainders = {x % 3 for x in range(10)}\nprint(remainders)\n\n# Generator expression (memory efficient)\ngen = (x**2 for x in range(1000000))  # doesn't create list\nprint(next(gen))\nprint(next(gen))\n\n# Nested comprehension\nmatrix = [[i*j for j in range(3)] for i in range(3)]\nprint(matrix)\n\n# Flatten nested list\nnested = [[1, 2], [3, 4], [5, 6]]\nflat = [item for sublist in nested for item in sublist]\nprint(flat)",
        "try_demo": None,
        "prerequisites": ["py_collections"]
    })
    
    items.append({
        "id": "py_files_io",
        "book_id": "fundamentals",
        "level": "intermediate",
        "title": "File I/O and Context Managers",
        "learn": "open(), read(), write(), with statement (context manager). File modes: 'r', 'w', 'a', 'rb', 'wb'. Automatic resource cleanup with 'with'.",
        "try_code": "# Writing to file\nwith open('example.txt', 'w') as f:\n    f.write('Hello, World!\\n')\n    f.write('Python File I/O\\n')\n\n# Reading from file\nwith open('example.txt', 'r') as f:\n    content = f.read()\n    print(content)\n\n# Read lines\nwith open('example.txt', 'r') as f:\n    for line in f:\n        print(line.strip())\n\n# Append to file\nwith open('example.txt', 'a') as f:\n    f.write('Appended line\\n')\n\n# Read/write binary\nimport pickle\ndata = {'name': 'Alice', 'scores': [95, 87, 92]}\nwith open('data.pkl', 'wb') as f:\n    pickle.dump(data, f)\n\nwith open('data.pkl', 'rb') as f:\n    loaded = pickle.load(f)\n    print(loaded)",
        "try_demo": None,
        "prerequisites": ["py_basics"]
    })
    
    items.append({
        "id": "py_exceptions",
        "book_id": "fundamentals",
        "level": "intermediate",
        "title": "Exception Handling: Try, Except, Finally",
        "learn": "try/except/else/finally, raise, custom exceptions. Graceful error handling. finally always executes (cleanup). Use specific exception types.",
        "try_code": "# Basic exception handling\ntry:\n    x = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')\n\n# Multiple exceptions\ntry:\n    result = int('not a number')\nexcept (ValueError, TypeError) as e:\n    print(f'Conversion error: {e}')\n\n# try/except/else/finally\ntry:\n    f = open('data.txt', 'r')\nexcept FileNotFoundError:\n    print('File not found')\nelse:\n    print('File opened successfully')\n    f.close()\nfinally:\n    print('Cleanup complete')\n\n# Raising exceptions\ndef validate_age(age):\n    if age < 0:\n        raise ValueError('Age cannot be negative')\n    return age\n\ntry:\n    validate_age(-5)\nexcept ValueError as e:\n    print(f'Validation error: {e}')",
        "try_demo": None,
        "prerequisites": ["py_functions"]
    })
    
    # ============================================================================
    # OBJECT-ORIENTED PROGRAMMING (5 items)
    # ============================================================================
    
    items.append({
        "id": "py_classes",
        "book_id": "oop",
        "level": "intermediate",
        "title": "Classes and Objects",
        "learn": "class, __init__, self, instance/class variables, methods. OOP encapsulates data and behavior. self is first parameter of instance methods.",
        "try_code": "# Define a class\nclass Student:\n    # Class variable (shared by all instances)\n    school = 'Python University'\n    \n    def __init__(self, name, age):\n        # Instance variables (unique to each instance)\n        self.name = name\n        self.age = age\n        self.grades = []\n    \n    def add_grade(self, grade):\n        self.grades.append(grade)\n    \n    def average(self):\n        if not self.grades:\n            return 0\n        return sum(self.grades) / len(self.grades)\n    \n    def __str__(self):\n        return f'Student({self.name}, {self.age})'\n\n# Create instances\nalice = Student('Alice', 20)\nalice.add_grade(95)\nalice.add_grade(87)\nprint(alice)\nprint(f'Average: {alice.average()}')\nprint(f'School: {Student.school}')",
        "try_demo": None,
        "prerequisites": ["py_functions"]
    })
    
    items.append({
        "id": "py_inheritance",
        "book_id": "oop",
        "level": "intermediate",
        "title": "Inheritance and Polymorphism",
        "learn": "Inheritance: class Child(Parent). super() calls parent methods. Polymorphism: different classes implement same interface. Method overriding.",
        "try_code": "# Base class\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n    \n    def speak(self):\n        return 'Some sound'\n    \n    def info(self):\n        return f'{self.name} says {self.speak()}'\n\n# Inheritance\nclass Dog(Animal):\n    def speak(self):\n        return 'Woof!'\n    \n    def fetch(self):\n        return f'{self.name} fetches the ball'\n\nclass Cat(Animal):\n    def speak(self):\n        return 'Meow!'\n\n# Polymorphism\nanimals = [Dog('Buddy'), Cat('Whiskers'), Animal('Unknown')]\nfor animal in animals:\n    print(animal.info())\n\n# super() for calling parent methods\nclass GoldenRetriever(Dog):\n    def __init__(self, name, color):\n        super().__init__(name)\n        self.color = color",
        "try_demo": None,
        "prerequisites": ["py_classes"]
    })
    
    items.append({
        "id": "py_magic_methods",
        "book_id": "oop",
        "level": "advanced",
        "title": "Magic Methods (Dunder Methods)",
        "learn": "__init__, __str__, __repr__, __len__, __getitem__, __add__, __eq__, __lt__. Customize object behavior. Operator overloading.",
        "try_code": "class Vector:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    \n    def __str__(self):\n        return f'Vector({self.x}, {self.y})'\n    \n    def __repr__(self):\n        return f'Vector(x={self.x}, y={self.y})'\n    \n    def __add__(self, other):\n        return Vector(self.x + other.x, self.y + other.y)\n    \n    def __mul__(self, scalar):\n        return Vector(self.x * scalar, self.y * scalar)\n    \n    def __eq__(self, other):\n        return self.x == other.x and self.y == other.y\n    \n    def __len__(self):\n        return int((self.x**2 + self.y**2)**0.5)\n\nv1 = Vector(3, 4)\nv2 = Vector(1, 2)\nprint(v1 + v2)\nprint(v1 * 2)\nprint(len(v1))\nprint(v1 == v2)",
        "try_demo": None,
        "prerequisites": ["py_inheritance"]
    })
    
    items.append({
        "id": "py_properties",
        "book_id": "oop",
        "level": "advanced",
        "title": "Properties and Encapsulation",
        "learn": "@property decorator for getters, @setter for setters, @deleter for deleters. Encapsulation: private attributes (_attr, __attr). Pythonic access control.",
        "try_code": "class Temperature:\n    def __init__(self, celsius):\n        self._celsius = celsius\n    \n    @property\n    def celsius(self):\n        return self._celsius\n    \n    @celsius.setter\n    def celsius(self, value):\n        if value < -273.15:\n            raise ValueError('Temperature below absolute zero')\n        self._celsius = value\n    \n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32\n    \n    @fahrenheit.setter\n    def fahrenheit(self, value):\n        self.celsius = (value - 32) * 5/9\n\ntemp = Temperature(25)\nprint(f'{temp.celsius}°C = {temp.fahrenheit}°F')\n\ntemp.fahrenheit = 98.6\nprint(f'{temp.celsius}°C = {temp.fahrenheit}°F')\n\n# Private attributes\nclass BankAccount:\n    def __init__(self, balance):\n        self.__balance = balance  # name mangling: _BankAccount__balance\n    \n    def deposit(self, amount):\n        if amount > 0:\n            self.__balance += amount",
        "try_demo": None,
        "prerequisites": ["py_magic_methods"]
    })
    
    items.append({
        "id": "py_dataclasses",
        "book_id": "oop",
        "level": "advanced",
        "title": "Dataclasses and Type Hints",
        "learn": "@dataclass decorator (Python 3.7+), type hints (: int, -> str), automatic __init__, __repr__, __eq__. Clean data containers.",
        "try_code": "from dataclasses import dataclass, field\nfrom typing import List, Optional\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n    \n    def distance(self) -> float:\n        return (self.x**2 + self.y**2)**0.5\n\n@dataclass\nclass Student:\n    name: str\n    age: int\n    grades: List[float] = field(default_factory=list)\n    advisor: Optional[str] = None\n    \n    def average(self) -> float:\n        if not self.grades:\n            return 0.0\n        return sum(self.grades) / len(self.grades)\n\np = Point(3.0, 4.0)\nprint(p)\nprint(f'Distance: {p.distance()}')\n\ns = Student('Alice', 20)\ns.grades.extend([95.5, 87.0, 92.5])\nprint(s)\nprint(f'Average: {s.average()}')\n\n# Type hints for functions\ndef greet(name: str, times: int = 1) -> str:\n    return (name + '! ') * times",
        "try_demo": None,
        "prerequisites": ["py_properties"]
    })
    
    # ============================================================================
    # FUNCTIONAL PROGRAMMING (5 items)
    # ============================================================================
    
    items.append({
        "id": "py_lambdas",
        "book_id": "functional",
        "level": "intermediate",
        "title": "Lambda Functions and Anonymous Functions",
        "learn": "lambda args: expression. Anonymous single-expression functions. Use for simple operations, sorting keys, map/filter.",
        "try_code": "# Lambda basics\nsquare = lambda x: x**2\nprint(square(5))\n\n# Sorting with lambda\nstudents = [('Alice', 95), ('Bob', 87), ('Charlie', 92)]\nstudents_by_score = sorted(students, key=lambda x: x[1], reverse=True)\nprint(students_by_score)\n\n# Multiple arguments\nadd = lambda x, y: x + y\nprint(add(3, 4))\n\n# Lambda in list comprehension key\nwords = ['Python', 'is', 'awesome', 'and', 'powerful']\nlongest = max(words, key=lambda w: len(w))\nprint(longest)\n\n# Conditional lambda\nabsolute = lambda x: x if x >= 0 else -x\nprint(absolute(-5))",
        "try_demo": None,
        "prerequisites": ["py_functions"]
    })
    
    items.append({
        "id": "py_map_filter",
        "book_id": "functional",
        "level": "intermediate",
        "title": "Map, Filter, Reduce",
        "learn": "map(func, iterable) applies func to each element. filter(func, iterable) keeps elements where func returns True. reduce(func, iterable) accumulates.",
        "try_code": "from functools import reduce\n\n# map: apply function to each element\nnums = [1, 2, 3, 4, 5]\nsquares = list(map(lambda x: x**2, nums))\nprint(squares)\n\n# Multiple iterables\na = [1, 2, 3]\nb = [10, 20, 30]\nsums = list(map(lambda x, y: x + y, a, b))\nprint(sums)\n\n# filter: keep elements matching condition\nevens = list(filter(lambda x: x % 2 == 0, nums))\nprint(evens)\n\n# reduce: accumulate values\nproduct = reduce(lambda x, y: x * y, nums)\nprint(product)\n\n# Comparison to comprehensions\nsquares_comp = [x**2 for x in nums]\nevens_comp = [x for x in nums if x % 2 == 0]\n# Comprehensions often more Pythonic!",
        "try_demo": None,
        "prerequisites": ["py_lambdas"]
    })
    
    items.append({
        "id": "py_decorators",
        "book_id": "functional",
        "level": "advanced",
        "title": "Decorators: Function Wrappers",
        "learn": "@decorator syntax. Decorators wrap functions to add behavior. Use for logging, timing, caching, validation. Higher-order functions.",
        "try_code": "import time\nfrom functools import wraps\n\n# Simple decorator\ndef timer(func):\n    @wraps(func)  # preserves func metadata\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end-start:.4f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(0.1)\n    return 'Done'\n\nslow_function()\n\n# Decorator with arguments\ndef repeat(times):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            for _ in range(times):\n                result = func(*args, **kwargs)\n            return result\n        return wrapper\n    return decorator\n\n@repeat(3)\ndef greet(name):\n    print(f'Hello, {name}!')\n\ngreet('Alice')",
        "try_demo": None,
        "prerequisites": ["py_map_filter"]
    })
    
    items.append({
        "id": "py_generators",
        "book_id": "functional",
        "level": "advanced",
        "title": "Generators and Yield",
        "learn": "yield produces values one at a time. Lazy evaluation: generates values on-demand. Memory efficient for large sequences. next() to get values.",
        "try_code": "# Generator function\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\n# Use generator\nfor num in fibonacci(10):\n    print(num, end=' ')\nprint()\n\n# Generator expression (like list comprehension with ())\ngen = (x**2 for x in range(1000000))  # doesn't create list!\nprint(next(gen))\nprint(next(gen))\n\n# Generator for infinite sequences\ndef count_up(start=0):\n    while True:\n        yield start\n        start += 1\n\ncounter = count_up(10)\nprint(next(counter))\nprint(next(counter))\n\n# Generator pipeline\nnums = range(100)\nevens = (x for x in nums if x % 2 == 0)\nsquares = (x**2 for x in evens)\nprint(list(squares)[:5])",
        "try_demo": None,
        "prerequisites": ["py_decorators"]
    })
    
    items.append({
        "id": "py_itertools",
        "book_id": "functional",
        "level": "advanced",
        "title": "Itertools: Functional Iterators",
        "learn": "itertools module: combinations, permutations, product, chain, cycle, islice, groupby. Efficient functional programming tools.",
        "try_code": "from itertools import combinations, permutations, product, chain, cycle, islice, groupby\n\n# Combinations\nletters = ['A', 'B', 'C']\nprint(list(combinations(letters, 2)))\n\n# Permutations\nprint(list(permutations(letters, 2)))\n\n# Cartesian product\nprint(list(product([1, 2], ['a', 'b'])))\n\n# Chain: flatten iterables\nprint(list(chain([1, 2], [3, 4], [5, 6])))\n\n# Cycle: infinite repetition (use with care!)\ncycler = cycle(['A', 'B', 'C'])\nprint(list(islice(cycler, 7)))  # first 7 elements\n\n# Groupby: group consecutive elements\ndata = [1, 1, 2, 2, 2, 3, 1, 1]\nfor key, group in groupby(data):\n    print(f'{key}: {list(group)}')\n\n# islice: slice an iterator\nfrom itertools import count\nfirst_10_evens = islice(count(0, 2), 10)\nprint(list(first_10_evens))",
        "try_demo": None,
        "prerequisites": ["py_generators"]
    })
    
    # ============================================================================
    # DATA PROCESSING (6 items)
    # ============================================================================
    
    items.append({
        "id": "py_numpy_basics",
        "book_id": "data_processing",
        "level": "intermediate",
        "title": "NumPy Arrays and Operations",
        "learn": "numpy.array, shape, dtype, indexing, slicing, broadcasting. Vectorized operations: fast element-wise math. Foundation for scientific Python.",
        "try_code": "import numpy as np\n\n# Create arrays\na = np.array([1, 2, 3, 4, 5])\nb = np.array([[1, 2, 3], [4, 5, 6]])\n\nprint(f'Shape: {b.shape}')  # (2, 3)\nprint(f'Dtype: {a.dtype}')  # int64\n\n# Vectorized operations\nprint(a * 2)  # element-wise\nprint(a ** 2)\nprint(np.sqrt(a))\n\n# Array creation\nzeros = np.zeros((3, 3))\nones = np.ones((2, 4))\nrange_arr = np.arange(0, 10, 2)\nlinspace = np.linspace(0, 1, 5)\n\n# Indexing and slicing\nprint(b[0, :])  # first row\nprint(b[:, 1])  # second column\n\n# Broadcasting\nmatrix = np.array([[1, 2], [3, 4]])\nvec = np.array([10, 20])\nprint(matrix + vec)  # adds vec to each row",
        "try_demo": None,
        "prerequisites": ["py_collections"]
    })
    
    items.append({
        "id": "py_numpy_advanced",
        "book_id": "data_processing",
        "level": "advanced",
        "title": "Advanced NumPy: Aggregations, Linear Algebra",
        "learn": "sum, mean, std, min, max, argmin, argmax along axes. dot product, matrix multiply (@), transpose, inverse, eigenvalues. reshape, flatten.",
        "try_code": "import numpy as np\n\ndata = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n\n# Aggregations\nprint(f'Sum: {np.sum(data)}')\nprint(f'Mean: {np.mean(data)}')\nprint(f'Std: {np.std(data)}')\nprint(f'Max: {np.max(data)}')\n\n# Along axis\nprint(f'Row sums: {np.sum(data, axis=1)}')\nprint(f'Col means: {np.mean(data, axis=0)}')\n\n# Linear algebra\nA = np.array([[1, 2], [3, 4]])\nB = np.array([[5, 6], [7, 8]])\n\nprint(f'Matrix multiply: {A @ B}')\nprint(f'Transpose: {A.T}')\nprint(f'Inverse: {np.linalg.inv(A)}')\nprint(f'Determinant: {np.linalg.det(A)}')\n\neigenvalues, eigenvectors = np.linalg.eig(A)\nprint(f'Eigenvalues: {eigenvalues}')\n\n# Reshape\nflat = data.flatten()\nreshaped = flat.reshape(3, 3)",
        "try_demo": None,
        "prerequisites": ["py_numpy_basics"]
    })
    
    items.append({
        "id": "py_pandas_basics",
        "book_id": "data_processing",
        "level": "intermediate",
        "title": "Pandas DataFrames and Series",
        "learn": "DataFrame (2D table), Series (1D array). Reading CSV, indexing (loc, iloc), columns, head, tail, describe, info.",
        "try_code": "import pandas as pd\nimport numpy as np\n\n# Create DataFrame\ndata = {\n    'Name': ['Alice', 'Bob', 'Charlie', 'David'],\n    'Age': [25, 30, 35, 40],\n    'Score': [95.5, 87.0, 92.5, 88.0]\n}\ndf = pd.DataFrame(data)\n\nprint(df)\nprint(f'\\nShape: {df.shape}')\nprint(f'\\nColumns: {df.columns}')\n\n# Indexing\nprint(df['Name'])  # Series\nprint(df[['Name', 'Score']])  # DataFrame\nprint(df.loc[0])  # row by label\nprint(df.iloc[1:3])  # rows by position\n\n# Statistics\nprint(df.describe())\nprint(df['Score'].mean())\n\n# Add column\ndf['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B')\nprint(df)",
        "try_demo": None,
        "prerequisites": ["py_numpy_basics"]
    })
    
    items.append({
        "id": "py_pandas_operations",
        "book_id": "data_processing",
        "level": "advanced",
        "title": "Pandas: Filtering, Groupby, Merge",
        "learn": "Boolean indexing, query(), groupby(), agg(), merge(), concat(), pivot_table(). Data manipulation and aggregation.",
        "try_code": "import pandas as pd\n\ndata = {\n    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice', 'Bob'],\n    'Subject': ['Math', 'Math', 'Math', 'Science', 'Science', 'Science'],\n    'Score': [95, 87, 92, 88, 91, 85]\n}\ndf = pd.DataFrame(data)\n\n# Filtering\nprint(df[df['Score'] > 90])\nprint(df.query('Score > 90 and Subject == \"Math\"'))\n\n# Groupby\nby_subject = df.groupby('Subject')['Score'].mean()\nprint(by_subject)\n\nmulti_group = df.groupby(['Name', 'Subject']).agg({'Score': ['mean', 'max']})\nprint(multi_group)\n\n# Merge\ndf1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})\ndf2 = pd.DataFrame({'ID': [1, 2], 'Age': [25, 30]})\nmerged = pd.merge(df1, df2, on='ID')\nprint(merged)\n\n# Pivot table\npivot = df.pivot_table(values='Score', index='Name', columns='Subject', aggfunc='mean')\nprint(pivot)",
        "try_demo": None,
        "prerequisites": ["py_pandas_basics"]
    })
    
    items.append({
        "id": "py_regex",
        "book_id": "data_processing",
        "level": "advanced",
        "title": "Regular Expressions",
        "learn": "re module: search, match, findall, sub. Patterns: ., *, +, ?, [], ^, $, \\d, \\w, \\s. Groups: (). Powerful text processing.",
        "try_code": "import re\n\ntext = 'Contact us at info@example.com or support@test.org'\n\n# Find email addresses\nemails = re.findall(r'\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b', text)\nprint(emails)\n\n# Search for pattern\nmatch = re.search(r'(\\w+)@(\\w+)\\.(\\w+)', text)\nif match:\n    print(f'User: {match.group(1)}')\n    print(f'Domain: {match.group(2)}')\n    print(f'TLD: {match.group(3)}')\n\n# Replace pattern\ncensored = re.sub(r'\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b', '[EMAIL]', text)\nprint(censored)\n\n# Split by pattern\ndata = 'apple,banana;orange|grape'\nfruits = re.split(r'[,;|]', data)\nprint(fruits)\n\n# Validate phone number\nphone = '123-456-7890'\nif re.match(r'^\\d{3}-\\d{3}-\\d{4}$', phone):\n    print('Valid phone number')",
        "try_demo": None,
        "prerequisites": ["py_strings"]
    })
    
    items.append({
        "id": "py_json_api",
        "book_id": "data_processing",
        "level": "intermediate",
        "title": "JSON and API Interaction",
        "learn": "json.dumps, json.loads, working with APIs. requests library: GET, POST, headers, params. Parse and process JSON data.",
        "try_code": "import json\n\n# JSON basics\ndata = {'name': 'Alice', 'age': 25, 'scores': [95, 87, 92]}\n\n# Serialize to JSON string\njson_str = json.dumps(data, indent=2)\nprint(json_str)\n\n# Deserialize from JSON\nparsed = json.loads(json_str)\nprint(parsed['name'])\n\n# File I/O\nwith open('data.json', 'w') as f:\n    json.dump(data, f, indent=2)\n\nwith open('data.json', 'r') as f:\n    loaded = json.load(f)\n\n# API example (requires requests library)\ntry:\n    import requests\n    \n    # GET request\n    response = requests.get('https://api.github.com/users/python')\n    if response.status_code == 200:\n        user_data = response.json()\n        print(f\"User: {user_data['login']}\")\nexcept ImportError:\n    print('Install requests: pip install requests')",
        "try_demo": None,
        "prerequisites": ["py_files_io"]
    })
    
    # ============================================================================
    # TESTING AND BEST PRACTICES (5 items)
    # ============================================================================
    
    items.append({
        "id": "py_testing",
        "book_id": "best_practices",
        "level": "intermediate",
        "title": "Unit Testing with pytest",
        "learn": "pytest for unit tests. Test functions start with test_. assert statements. Fixtures for setup. Run: pytest file.py. Test-driven development.",
        "try_code": "# File: test_math_ops.py\nimport pytest\n\ndef add(a, b):\n    return a + b\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n\n# Test functions\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n    assert add(0, 0) == 0\n\ndef test_divide():\n    assert divide(10, 2) == 5\n    assert divide(9, 3) == 3\n\ndef test_divide_by_zero():\n    with pytest.raises(ValueError):\n        divide(10, 0)\n\n# Fixture for setup\n@pytest.fixture\ndef sample_data():\n    return [1, 2, 3, 4, 5]\n\ndef test_with_fixture(sample_data):\n    assert len(sample_data) == 5\n    assert sum(sample_data) == 15\n\n# Run: pytest test_math_ops.py",
        "try_demo": None,
        "prerequisites": ["py_exceptions"]
    })
    
    items.append({
        "id": "py_debugging",
        "book_id": "best_practices",
        "level": "intermediate",
        "title": "Debugging: pdb and Logging",
        "learn": "pdb (Python debugger): breakpoint(), n (next), c (continue), p (print). logging module: info, warning, error. Debug efficiently.",
        "try_code": "import logging\n\n# Setup logging\nlogging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')\n\ndef complex_calculation(x):\n    logging.info(f'Starting calculation with x={x}')\n    \n    if x < 0:\n        logging.warning('Negative input detected')\n    \n    try:\n        result = 100 / x\n        logging.info(f'Result: {result}')\n        return result\n    except ZeroDivisionError:\n        logging.error('Division by zero')\n        return None\n\ncomplex_calculation(10)\ncomplex_calculation(0)\n\n# Debugging with pdb\ndef buggy_function(n):\n    total = 0\n    for i in range(n):\n        # breakpoint()  # Uncomment to start debugger\n        total += i\n    return total\n\n# In debugger:\n# n - next line\n# s - step into\n# c - continue\n# p variable - print variable\n# l - list code",
        "try_demo": None,
        "prerequisites": ["py_exceptions"]
    })
    
    items.append({
        "id": "py_typing",
        "book_id": "best_practices",
        "level": "advanced",
        "title": "Type Hints and Mypy",
        "learn": "Type annotations: def func(x: int) -> str. typing module: List, Dict, Optional, Union, Callable. mypy for static type checking.",
        "try_code": "from typing import List, Dict, Optional, Union, Callable, Tuple\n\n# Basic type hints\ndef greet(name: str, times: int = 1) -> str:\n    return (name + '! ') * times\n\n# Collections\ndef process_scores(scores: List[float]) -> float:\n    if not scores:\n        return 0.0\n    return sum(scores) / len(scores)\n\n# Optional (can be None)\ndef find_user(user_id: int) -> Optional[str]:\n    users = {1: 'Alice', 2: 'Bob'}\n    return users.get(user_id)\n\n# Union (multiple types)\ndef convert(value: Union[int, float, str]) -> float:\n    return float(value)\n\n# Callable (function type)\ndef apply_operation(x: int, op: Callable[[int], int]) -> int:\n    return op(x)\n\n# Type alias\nVector = List[float]\nMatrix = List[Vector]\n\ndef dot_product(v1: Vector, v2: Vector) -> float:\n    return sum(a * b for a, b in zip(v1, v2))\n\n# Run mypy: mypy script.py",
        "try_demo": None,
        "prerequisites": ["py_dataclasses"]
    })
    
    items.append({
        "id": "py_context_managers",
        "book_id": "best_practices",
        "level": "advanced",
        "title": "Custom Context Managers",
        "learn": "__enter__ and __exit__ methods. contextlib.contextmanager decorator. Ensure resource cleanup. Use with 'with' statement.",
        "try_code": "from contextlib import contextmanager\nimport time\n\n# Class-based context manager\nclass Timer:\n    def __enter__(self):\n        self.start = time.time()\n        return self\n    \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.end = time.time()\n        self.elapsed = self.end - self.start\n        print(f'Elapsed: {self.elapsed:.4f}s')\n        return False  # Don't suppress exceptions\n\nwith Timer():\n    sum(range(1000000))\n\n# Decorator-based context manager\n@contextmanager\ndef timing(label):\n    start = time.time()\n    try:\n        yield\n    finally:\n        end = time.time()\n        print(f'{label}: {end-start:.4f}s')\n\nwith timing('Calculation'):\n    result = sum(range(1000000))\n\n# File-like context manager\n@contextmanager\ndef managed_file(filename, mode):\n    f = open(filename, mode)\n    try:\n        yield f\n    finally:\n        f.close()",
        "try_demo": None,
        "prerequisites": ["py_files_io"]
    })
    
    items.append({
        "id": "py_performance",
        "book_id": "best_practices",
        "level": "advanced",
        "title": "Performance: Profiling and Optimization",
        "learn": "timeit for micro-benchmarks, cProfile for profiling, memory_profiler. Use generators, NumPy, list comprehensions. Algorithmic optimization.",
        "try_code": "import timeit\nimport cProfile\nimport pstats\nfrom io import StringIO\n\n# timeit for micro-benchmarks\nsetup = 'nums = list(range(1000))'\nlist_comp = timeit.timeit('[x**2 for x in nums]', setup=setup, number=10000)\nmap_loop = timeit.timeit('list(map(lambda x: x**2, nums))', setup=setup, number=10000)\nprint(f'List comp: {list_comp:.4f}s')\nprint(f'Map: {map_loop:.4f}s')\n\n# cProfile for function profiling\ndef slow_function():\n    total = 0\n    for i in range(1000):\n        for j in range(1000):\n            total += i * j\n    return total\n\nprofiler = cProfile.Profile()\nprofiler.enable()\nslow_function()\nprofiler.disable()\n\n# Print stats\ns = StringIO()\nps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')\nps.print_stats(10)\n# print(s.getvalue())\n\n# Optimization tips:\n# 1. Use generators for large sequences\n# 2. Use NumPy for numeric operations\n# 3. Cache repeated calculations\n# 4. Choose right data structure (set for membership)",
        "try_demo": None,
        "prerequisites": ["py_decorators"]
    })
    
    return items


def main():
    """Generate and save enriched curriculum."""
    items = generate_curriculum_items()
    
    # Save to cache
    cache_dir = Path(__file__).parent.parent / '.cache'
    cache_dir.mkdir(exist_ok=True)
    
    output_file = cache_dir / 'python_practice_enriched.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    
    # Statistics
    by_level = {}
    by_book = {}
    for item in items:
        by_level[item['level']] = by_level.get(item['level'], 0) + 1
        by_book[item['book_id']] = by_book.get(item['book_id'], 0) + 1
    
    print(f"\n{'='*70}")
    print(f"✅ Generated {len(items)} curriculum items for Python Practice Lab")
    print(f"{'='*70}")
    print(f"\nBy Level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        print(f"  {level}: {by_level.get(level, 0)}")
    
    print(f"\nBy Topic:")
    for book_id, count in by_book.items():
        print(f"  {book_id}: {count}")
    
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
