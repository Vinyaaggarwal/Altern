import engine_v2

def test(label, code, src, tgt):
    result = engine_v2.full_translate(code.strip(), src, tgt)
    print(f'=== {label} ===')
    print(result)
    print()

# 1. Python if/else -> C++
test('Python if/else -> C++', """
x = 10
if x > 5:
    print('big')
else:
    print('small')
""", 'python', 'cpp')

# 2. Python if/elif/else -> Java
test('Python if/elif/else -> Java', """
x = 7
if x > 10:
    print('large')
elif x > 5:
    print('medium')
else:
    print('small')
""", 'python', 'java')

# 3. Python while -> JavaScript
test('Python while -> JavaScript', """
def countdown(n):
    while n > 0:
        print(n)
        n -= 1

countdown(5)
""", 'python', 'javascript')

# 4. C++ if/else -> Python
test('C++ if/else -> Python', """
#include <iostream>
using namespace std;
int main() {
    int x = 10;
    if (x > 5) {
        cout << "big" << endl;
    } else {
        cout << "small" << endl;
    }
    return 0;
}
""", 'cpp', 'python')

# 5. Java if/else -> Python
test('Java if/else -> Python', """
public class Main {
    public static void main(String[] args) {
        int x = 10;
        if (x > 5) {
            System.out.println("big");
        } else {
            System.out.println("small");
        }
    }
}
""", 'java', 'python')

# 6. Python -> TypeScript
test('Python func -> TypeScript', """
def add(a, b):
    return a + b

result = add(3, 4)
print(result)
""", 'python', 'typescript')

# 7. JavaScript -> C++
test('JavaScript -> C++', """
function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
let r = factorial(5);
console.log(r);
""", 'javascript', 'cpp')

# 8. C++ -> Java
test('C++ -> Java', """
#include <iostream>
using namespace std;
int add(int a, int b) {
    return a + b;
}
int main() {
    int x = add(3, 4);
    cout << x << endl;
    return 0;
}
""", 'cpp', 'java')

# 9. Java -> JavaScript
test('Java -> JavaScript', """
public class Main {
    public static int add(int a, int b) {
        return a + b;
    }
    public static void main(String[] args) {
        int x = add(3, 4);
        System.out.println(x);
    }
}
""", 'java', 'javascript')

# 10. TypeScript -> Python
test('TypeScript -> Python', """
function greet(name: string): string {
    return "Hello " + name;
}
let msg = greet("World");
console.log(msg);
""", 'typescript', 'python')
