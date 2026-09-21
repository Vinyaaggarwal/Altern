import sys
import os

sys.path.insert(0, r"c:\Users\vinya\OneDrive\Documents\major\Altern\model_server")
import server

VECTOR_SAMPLES = {
    "cpp": """#include <iostream>
#include <vector>
using namespace std;
int main() {
    vector<int> nums = {10, 20, 30};
    int sum = 0;
    for (int n : nums) {
        cout << n << endl;
        sum += n;
    }
    cout << sum << endl;
    return 0;
}""",
    "java": """import java.util.*;
public class Main {
    public static void main(String[] args) {
        int[] nums = {10, 20, 30};
        int sum = 0;
        for (int n : nums) {
            System.out.println(n);
            sum += n;
        }
        System.out.println(sum);
    }
}""",
    "python": """nums = [10, 20, 30]
sum = 0
for n in nums:
    print(n)
    sum += n
print(sum)""",
    "javascript": """const nums = [10, 20, 30];
let sum = 0;
for (const n of nums) {
    console.log(n);
    sum += n;
}
console.log(sum);""",
    "typescript": """const nums: number[] = [10, 20, 30];
let sum: number = 0;
for (const n of nums) {
    console.log(n);
    sum += n;
}
console.log(sum);"""
}

languages = ["cpp", "java", "python", "javascript", "typescript"]

print("=== TESTING VECTOR & ARITHMETIC LOOPS (20 PAIRS) ===")
failures = []
passed = 0
total = 0

for src in languages:
    for tgt in languages:
        if src == tgt:
            continue
        total += 1
        source_code = VECTOR_SAMPLES[src]
        translated = server.rule_assisted_fallback_translation(source_code, src, tgt)
        exec_res = server.run_code_execution(translated, tgt)
        
        ok = (exec_res['status'] == "success" and "60" in exec_res['stdout'])
        if ok:
            passed += 1
            print(f"  [PASS] {src.upper()} -> {tgt.upper()}")
        else:
            err = exec_res['stderr'] or exec_res['stdout'] or exec_res['status']
            failures.append((src, tgt, err, translated))
            print(f"  [FAIL] {src.upper()} -> {tgt.upper()}: {err[:120]}")

print("\n" + "="*50)
print(f"VECTOR LOOPS RESULT: {passed}/{total} EXECUTED AND PASSED")
if failures:
    for src, tgt, err, code in failures:
        print(f"  {src} -> {tgt}: {err[:100]}")
