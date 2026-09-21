"""
Verification & Automated Benchmark Suite for Local Code Translation Model
"""
import sys
from server import rule_assisted_fallback_translation

def run_translation_tests():
    print("==================================================")
    print("Running Altern Local Translation Verification")
    print("==================================================")

    test_cases = [
        {
            "name": "C++ to Java Vector Translation",
            "source_lang": "cpp",
            "target_lang": "java",
            "code": '#include <iostream>\n#include <vector>\n\nint main() {\n    std::vector<int> nums = {10, 20};\n    std::cout << nums[0] << std::endl;\n    return 0;\n}',
            "expected_keyword": "System.out.println"
        },
        {
            "name": "Java to Python Printing & Main Method",
            "source_lang": "java",
            "target_lang": "python",
            "code": 'public class Test {\n    public static void main(String[] args) {\n        System.out.println("Hello World");\n    }\n}',
            "expected_keyword": 'print("Hello World")'
        },
        {
            "name": "C++ to Python I/O Translation",
            "source_lang": "cpp",
            "target_lang": "python",
            "code": '#include <iostream>\n\nint main() {\n    std::cout << "Live Syntax" << std::endl;\n    return 0;\n}',
            "expected_keyword": 'print("Live Syntax")'
        },
        {
            "name": "Complex Multi-Line C++ Nested Loops to Python",
            "source_lang": "cpp",
            "target_lang": "python",
            "code": '''#include <iostream>
#include <vector>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int limit = 5;
    for (int i = 1; i <= limit; i++) {
        if (i % 2 == 0) {
            std::cout << i << " is even" << std::endl;
        } else {
            std::cout << i << " is odd" << std::endl;
        }
    }
    return 0;
}''',
            "expected_keyword": "def factorial(n):"
        }
    ]

    passed = 0
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] {test['name']} ({test['source_lang']} -> {test['target_lang']})")
        converted = rule_assisted_fallback_translation(test["code"], test["source_lang"], test["target_lang"])
        
        print("--- Output ---")
        print(converted)
        print("--------------")

        if test["expected_keyword"] in converted:
            print(f"[PASSED] (Found '{test['expected_keyword']}')")
            passed += 1
        else:
            print(f"[FAILED] (Missing '{test['expected_keyword']}')")

    print("\n==================================================")
    print(f"Test Summary: {passed}/{len(test_cases)} Tests Passed")
    print("==================================================")

    if passed == len(test_cases):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_translation_tests()

