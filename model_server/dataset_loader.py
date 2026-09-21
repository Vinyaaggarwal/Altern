"""
Dataset Loader for Code Translation Parallel Corpus
"""
import os
import json
from typing import List, Dict

# Sample Parallel Translation Corpus for Core Programming Languages
SAMPLE_PARALLEL_DATA: List[Dict[str, str]] = [
    {
        "source_lang": "cpp",
        "target_lang": "java",
        "source_code": '#include <iostream>\n#include <vector>\n\nint main() {\n    std::vector<int> numbers = {1, 2, 3, 4, 5};\n    for (int n : numbers) {\n        std::cout << n << std::endl;\n    }\n    return 0;\n}',
        "target_code": 'import java.util.ArrayList;\nimport java.util.Arrays;\nimport java.util.List;\n\npublic class Main {\n    public static void main(String[] args) {\n        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);\n        for (int n : numbers) {\n            System.out.println(n);\n        }\n    }\n}'
    },
    {
        "source_lang": "java",
        "target_lang": "python",
        "source_code": 'public class Calculator {\n    public static int add(int a, int b) {\n        return a + b;\n    }\n    public static void main(String[] args) {\n        System.out.println(add(5, 10));\n    }\n}',
        "target_code": 'def add(a: int, b: int) -> int:\n    return a + b\n\nif __name__ == "__main__":\n    print(add(5, 10))'
    },
    {
        "source_lang": "cpp",
        "target_lang": "python",
        "source_code": '#include <iostream>\n#include <string>\n\nstd::string greet(std::string name) {\n    return "Hello, " + name;\n}\n\nint main() {\n    std::cout << greet("Alice") << std::endl;\n    return 0;\n}',
        "target_code": 'def greet(name: str) -> str:\n    return f"Hello, {name}"\n\nif __name__ == "__main__":\n    print(greet("Alice"))'
    },
    {
        "source_lang": "javascript",
        "target_lang": "typescript",
        "source_code": 'function calculateTotal(items) {\n    let total = 0;\n    for (let item of items) {\n        total += item.price;\n    }\n    return total;\n}',
        "target_code": 'interface Item {\n    price: number;\n}\n\nfunction calculateTotal(items: Item[]): number {\n    let total: number = 0;\n    for (let item of items) {\n        total += item.price;\n    }\n    return total;\n}'
    }
]

def load_parallel_dataset(save_path: str = "parallel_dataset.json") -> List[Dict[str, str]]:
    """Loads parallel translation dataset or generates sample if not present."""
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Save default sample dataset
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_PARALLEL_DATA, f, indent=2)
    return SAMPLE_PARALLEL_DATA

if __name__ == "__main__":
    data = load_parallel_dataset()
    print(f"Loaded {len(data)} parallel code translation pairs.")
