"""
Enrich SICP Lab - Structure and Interpretation of Computer Programs
===================================================================

Generates curriculum items for functional programming concepts from SICP.
"""

from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_sicp_curriculum():
    """Generate SICP curriculum items covering key concepts."""
    
    items = [
        # Basics - Building Blocks
        {"id": "sicp_expressions", "book_id": "sicp", "level": "basics", "title": "Expressions and Combinations",
         "learn": "Programs built from expressions (primitives, combinations). Evaluation: apply operator to operands. Prefix notation: (* (+ 2 3) 4) = 20. Composition builds complexity from simple parts.",
         "try_code": "# Python equivalent of Scheme expressions\nresult = (2 + 3) * 4  # 20\nprint(f'Result: {result}')",
         "try_demo": "sicp_expressions"},
        
        {"id": "sicp_procedures", "book_id": "sicp", "level": "basics", "title": "Defining Procedures",
         "learn": "Procedures abstract computational patterns: (define (square x) (* x x)). Function definition creates abstraction. Parameters provide generality.",
         "try_code": "def square(x):\n    return x * x\n\nprint(square(5))  # 25",
         "try_demo": "sicp_procedures"},
        
        {"id": "sicp_substitution", "book_id": "sicp", "level": "basics", "title": "Substitution Model",
         "learn": "Evaluate by substituting arguments: (square 5) → (* 5 5) → 25. Two orders: normal (lazy) vs applicative (eager). Foundation for understanding evaluation.",
         "try_code": "# Applicative order (Python default)\n# Evaluate arguments first, then apply\ndef f(a, b):\n    return a + a\n# f(3+2, 4*5) → f(5, 20) → 5+5 → 10",
         "try_demo": None},
        
        # Intermediate - Higher-Order Procedures
        {"id": "sicp_higher_order", "book_id": "sicp", "level": "intermediate", "title": "Higher-Order Procedures",
         "learn": "Procedures as first-class: accept procedures as arguments, return procedures as values. Example: map, filter, reduce. Enables powerful abstractions.",
         "try_code": "from functools import reduce\n\n# map: apply function to each element\nsquared = list(map(lambda x: x**2, [1,2,3,4]))\n\n# filter: select elements\neven = list(filter(lambda x: x%2==0, [1,2,3,4]))\n\n# reduce: accumulate\nsum_all = reduce(lambda a,b: a+b, [1,2,3,4])",
         "try_demo": "sicp_higher_order"},
        
        {"id": "sicp_lambda", "book_id": "sicp", "level": "intermediate", "title": "Lambda Expressions",
         "learn": "Anonymous functions: (lambda (x) (* x x)). Create procedures without naming. Essential for higher-order programming. Closures capture environment.",
         "try_code": "# Lambda expressions\nsquare = lambda x: x * x\nadd = lambda x, y: x + y\n\n# Use in higher-order functions\nresult = list(map(lambda x: x**2, [1,2,3]))",
         "try_demo": "sicp_lambda"},
        
        {"id": "sicp_closures", "book_id": "sicp", "level": "intermediate", "title": "Closures and Environments",
         "learn": "Closure: procedure + environment where it was created. Inner functions capture outer variables. Enables data hiding, factories, decorators.",
         "try_code": "def make_counter():\n    count = 0\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment\n\ncounter = make_counter()\nprint(counter())  # 1\nprint(counter())  # 2",
         "try_demo": "sicp_closures"},
        
        {"id": "sicp_recursion", "book_id": "sicp", "level": "intermediate", "title": "Recursion and Iteration",
         "learn": "Recursive process: grows then shrinks (factorial). Iterative process: constant space (tail recursion). Linear vs tree recursion. Identify base case and recursive step.",
         "try_code": "# Recursive factorial\ndef factorial_recursive(n):\n    if n <= 1:\n        return 1\n    return n * factorial_recursive(n-1)\n\n# Iterative factorial\ndef factorial_iterative(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
         "try_demo": "sicp_recursion"},
        
        {"id": "sicp_tree_recursion", "book_id": "sicp", "level": "intermediate", "title": "Tree Recursion",
         "learn": "Multiple recursive calls: Fibonacci, tree traversal. Exponential time complexity. Often inefficient but expressive. Can optimize with memoization.",
         "try_code": "def fib_tree(n):\n    if n <= 1:\n        return n\n    return fib_tree(n-1) + fib_tree(n-2)\n\n# With memoization\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib_memo(n):\n    if n <= 1:\n        return n\n    return fib_memo(n-1) + fib_memo(n-2)",
         "try_demo": "sicp_tree_recursion"},
        
        # Advanced - Data Abstraction
        {"id": "sicp_data_abstraction", "book_id": "sicp", "level": "advanced", "title": "Data Abstraction",
         "learn": "Separate representation from use. Constructors and selectors. Example: rational numbers (make-rat, numer, denom). Abstraction barriers isolate implementation.",
         "try_code": "# Rational number abstraction\nclass Rational:\n    def __init__(self, numer, denom):\n        self.numer = numer\n        self.denom = denom\n    \n    def __add__(self, other):\n        return Rational(\n            self.numer * other.denom + other.numer * self.denom,\n            self.denom * other.denom\n        )\n    \n    def __str__(self):\n        return f'{self.numer}/{self.denom}'",
         "try_demo": "sicp_data_abstraction"},
        
        {"id": "sicp_sequences", "book_id": "sicp", "level": "advanced", "title": "Sequences and List Operations",
         "learn": "Lists as sequences: cons, car, cdr in Scheme. Map, filter, accumulate (reduce) as sequence operations. List comprehensions in Python. Uniform interface.",
         "try_code": "# Sequence operations\ndata = [1, 2, 3, 4, 5]\n\n# Map\nsquared = [x**2 for x in data]\n\n# Filter\neven = [x for x in data if x % 2 == 0]\n\n# Reduce (accumulate)\nfrom functools import reduce\nsum_all = reduce(lambda a,b: a+b, data)\n\n# Nested sequences\nmatrix = [[1,2], [3,4], [5,6]]\nflattened = [x for row in matrix for x in row]",
         "try_demo": "sicp_sequences"},
        
        {"id": "sicp_streams", "book_id": "sicp", "level": "advanced", "title": "Streams and Delayed Evaluation",
         "learn": "Infinite sequences via lazy evaluation: stream-cons delays cdr. Example: infinite stream of integers. Decouples order of events from program structure. Python generators.",
         "try_code": "# Python generators as streams\ndef integers_from(n):\n    while True:\n        yield n\n        n += 1\n\n# Infinite stream of integers\nintegers = integers_from(0)\n\n# Take first 10\nfirst_10 = [next(integers) for _ in range(10)]\n\n# Stream filtering\ndef sieve(stream):\n    first = next(stream)\n    yield first\n    yield from sieve(x for x in stream if x % first != 0)",
         "try_demo": "sicp_streams"},
        
        {"id": "sicp_metacircular", "book_id": "sicp", "level": "expert", "title": "Metacircular Evaluator",
         "learn": "Interpreter written in language it interprets. Eval-apply cycle: eval determines meaning of expressions, apply executes procedures. Reveals language implementation.",
         "try_code": "# Simple expression evaluator\ndef eval_expr(expr, env):\n    if isinstance(expr, (int, float)):\n        return expr\n    elif isinstance(expr, str):  # variable\n        return env[expr]\n    elif expr[0] == '+':\n        return eval_expr(expr[1], env) + eval_expr(expr[2], env)\n    elif expr[0] == '*':\n        return eval_expr(expr[1], env) * eval_expr(expr[2], env)\n\n# Example: (+ (* 2 3) 4)\nexpr = ['+', ['*', 2, 3], 4]\nresult = eval_expr(expr, {})",
         "try_demo": "sicp_metacircular"},
        
        # Advanced - Modularity and State
        {"id": "sicp_state", "book_id": "sicp", "level": "advanced", "title": "Assignment and Local State",
         "learn": "Introduce mutation: set! changes variable. Local state: encapsulate state in closure. Trade-off: power vs complexity. Environment model replaces substitution.",
         "try_code": "# Object with local state\nclass BankAccount:\n    def __init__(self, balance):\n        self._balance = balance\n    \n    def withdraw(self, amount):\n        if self._balance >= amount:\n            self._balance -= amount\n            return self._balance\n        return 'Insufficient funds'\n    \n    def deposit(self, amount):\n        self._balance += amount\n        return self._balance",
         "try_demo": None},
        
        {"id": "sicp_mutation", "book_id": "sicp", "level": "advanced", "title": "Mutation and Identity",
         "learn": "Mutation introduces time: same object, different values. Identity vs equality: (eq? vs equal?). Aliasing problems. Functional vs imperative style trade-offs.",
         "try_code": "# Mutation and aliasing\na = [1, 2, 3]\nb = a  # Aliasing\nb.append(4)\nprint(a)  # [1, 2, 3, 4] - mutated!\n\n# Avoid with copying\nc = [1, 2, 3]\nd = c.copy()\nd.append(4)\nprint(c)  # [1, 2, 3] - unchanged",
         "try_demo": None},
        
        {"id": "sicp_dispatch", "book_id": "sicp", "level": "advanced", "title": "Message Passing and Dispatch",
         "learn": "Objects as dispatch procedures: send messages to select operations. Example: (account 'withdraw 50). Basis for OOP. Data-directed programming.",
         "try_code": "# Message passing style\ndef make_account(balance):\n    def dispatch(message, *args):\n        if message == 'withdraw':\n            nonlocal balance\n            amount = args[0]\n            if balance >= amount:\n                balance -= amount\n                return balance\n            return 'Insufficient funds'\n        elif message == 'deposit':\n            nonlocal balance\n            balance += args[0]\n            return balance\n        elif message == 'balance':\n            return balance\n    return dispatch\n\naccount = make_account(100)\nprint(account('withdraw', 50))  # 50\nprint(account('balance'))  # 50",
         "try_demo": "sicp_dispatch"},
        
        # Expert - Advanced Topics
        {"id": "sicp_continuations", "book_id": "sicp", "level": "expert", "title": "Continuations and Control",
         "learn": "Continuation: rest of computation. call/cc captures continuation as first-class value. Enables non-local exits, backtracking, coroutines. Powerful but complex.",
         "try_code": "# Python approximation with exceptions\nclass Escape(Exception):\n    def __init__(self, value):\n        self.value = value\n\ndef call_with_escape(func):\n    def escape(value):\n        raise Escape(value)\n    try:\n        return func(escape)\n    except Escape as e:\n        return e.value\n\n# Use for early return\nresult = call_with_escape(lambda k: k(42) if True else 100)",
         "try_demo": None},
        
        {"id": "sicp_amb", "book_id": "sicp", "level": "expert", "title": "Nondeterministic Computing",
         "learn": "amb operator: choose among alternatives, backtrack on failure. Declarative programming: specify what, not how. Logic puzzles, constraint satisfaction.",
         "try_code": "# Backtracking search approximation\ndef solve_puzzle():\n    for a in range(1, 10):\n        for b in range(1, 10):\n            for c in range(1, 10):\n                if a + b + c == 15 and a < b < c:\n                    yield (a, b, c)\n\nsolutions = list(solve_puzzle())\nprint(solutions)",
         "try_demo": None},
        
        {"id": "sicp_register_machine", "book_id": "sicp", "level": "expert", "title": "Register Machines",
         "learn": "Low-level machine model: registers, operations, controller. Compile high-level to register operations. Bridge between abstraction and hardware.",
         "try_code": "# Simulated register machine for GCD\nclass RegisterMachine:\n    def __init__(self):\n        self.registers = {'a': 0, 'b': 0, 't': 0}\n    \n    def gcd(self, a, b):\n        self.registers['a'] = a\n        self.registers['b'] = b\n        while self.registers['b'] != 0:\n            self.registers['t'] = self.registers['b']\n            self.registers['b'] = self.registers['a'] % self.registers['b']\n            self.registers['a'] = self.registers['t']\n        return self.registers['a']",
         "try_demo": None},
        
        # Practical Applications
        {"id": "sicp_symbolic", "book_id": "sicp", "level": "advanced", "title": "Symbolic Data and Differentiation",
         "learn": "Represent expressions as data: (+ x 3) is a list. Symbolic differentiation: d(x^n)/dx = n*x^(n-1). Pattern matching and rewriting.",
         "try_code": "# Symbolic differentiation\ndef deriv(expr, var):\n    if isinstance(expr, (int, float)):\n        return 0\n    elif expr == var:\n        return 1\n    elif expr[0] == '+':\n        return ['+', deriv(expr[1], var), deriv(expr[2], var)]\n    elif expr[0] == '*':\n        # Product rule: d(uv) = u*dv + v*du\n        u, v = expr[1], expr[2]\n        return ['+', ['*', u, deriv(v, var)], ['*', v, deriv(u, var)]]\n\n# Example: d(x^2)/dx\nexpr = ['*', 'x', 'x']\nprint(deriv(expr, 'x'))",
         "try_demo": "sicp_symbolic"},
        
        {"id": "sicp_interpreter_design", "book_id": "sicp", "level": "expert", "title": "Language Design Principles",
         "learn": "Separate syntax (parsing) from semantics (evaluation). Primitive expressions, means of combination, means of abstraction. Extensibility via new syntax and special forms.",
         "try_code": "# Extensible expression evaluator\nclass Evaluator:\n    def __init__(self):\n        self.forms = {\n            '+': lambda args, env: sum(self.eval(a, env) for a in args),\n            '*': lambda args, env: eval('*'.join(str(self.eval(a, env)) for a in args)),\n        }\n    \n    def eval(self, expr, env={}):\n        if isinstance(expr, (int, float)):\n            return expr\n        return self.forms[expr[0]](expr[1:], env)",
         "try_demo": None},
    ]
    
    return items


def main():
    print("=" * 70)
    print("SICP LAB ENRICHMENT")
    print("=" * 70)
    print()
    
    items = generate_sicp_curriculum()
    
    print(f"✅ Generated {len(items)} curriculum items")
    
    # Distribution
    from collections import Counter
    level_counts = Counter(item['level'] for item in items)
    print(f"\nBy level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        count = level_counts.get(level, 0)
        print(f"  {level:15s}: {count}")
    
    # Save
    output_dir = REPO_ROOT / "learning_apps" / ".cache"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "sicp_enriched.json"
    output_file.write_text(json.dumps(items, indent=2))
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"New items: {len(items)}")
    print(f"Target achieved: 20+ items covering SICP chapters 1-5")
    print(f"\nTopics:")
    print(f"  • Expressions & Procedures (basics)")
    print(f"  • Higher-Order Functions & Recursion (intermediate)")
    print(f"  • Data Abstraction & Streams (advanced)")
    print(f"  • Metacircular Evaluator & Continuations (expert)")


if __name__ == "__main__":
    main()
