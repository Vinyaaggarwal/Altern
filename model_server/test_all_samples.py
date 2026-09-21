import sys
import os

sys.path.insert(0, r"c:\Users\vinya\OneDrive\Documents\major\Altern\model_server")
import server

SAMPLES = {
    "cpp": """#include <iostream>
#include <vector>

int main() {
    std::vector<int> nums = {10, 20, 30};
    for (int n : nums) {
        std::cout << n << std::endl;
    }
    return 0;
}""",
    "java": """import java.util.*;

public class Main {
    public static void main(String[] args) {
        List<Integer> nums = Arrays.asList(10, 20, 30);
        for (int n : nums) {
            System.out.println(n);
        }
    }
}""",
    "python": """def greet(name: str) -> str:
    return f"Hello, {name}!"

names = ["Alice", "Bob", "Charlie"]
for name in names:
    print(greet(name))""",
    "javascript": """const nums = [10, 20, 30];

function double(n) {
    return n * 2;
}

for (let i = 0; i < nums.length; i++) {
    console.log(double(nums[i]));
}""",
    "typescript": """interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

const user: User = { name: "Alice", age: 30 };
console.log(greet(user));"""
}

languages = ["cpp", "java", "python", "javascript", "typescript"]

print("=== TESTING SERVER TRANSLATION & EXECUTION (20 PAIRS) ===")
failures = []
passed = 0
total = 0

for src in languages:
    for tgt in languages:
        if src == tgt:
            continue
        total += 1
        source_code = SAMPLES[src]
        translated = server.rule_assisted_fallback_translation(source_code, src, tgt)
        
        exec_res = server.run_code_execution(translated, tgt)
        if exec_res['status'] == "success":
            passed += 1
            print(f"  [PASS] {src.upper()} -> {tgt.upper()} (stdout: {repr(exec_res['stdout'].strip()[:30])})")
        else:
            err = exec_res['stderr'] or exec_res['stdout'] or exec_res['status']
            failures.append((src, tgt, err, translated))
            print(f"  [FAIL] {src.upper()} -> {tgt.upper()}: {err[:120]}")

print("\n" + "="*50)
print(f"FINAL RESULT: {passed}/{total} EXECUTED AND PASSED")
if failures:
    print("\nFAILURES:")
    for src, tgt, err, code in failures:
        print(f"  {src} -> {tgt}: {err[:150]}")
        print("CODE:\n" + code + "\n")
