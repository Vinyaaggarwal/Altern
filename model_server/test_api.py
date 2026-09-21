import requests

test_cases = [
    {
        "name": "C++ -> Python",
        "body": {
            "source_code": '#include <iostream>\nint main() {\n    std::cout << "Hello World" << std::endl;\n    return 0;\n}',
            "source_lang": "cpp",
            "target_lang": "python"
        }
    },
    {
        "name": "Java -> Python",
        "body": {
            "source_code": 'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello World");\n    }\n}',
            "source_lang": "java",
            "target_lang": "python"
        }
    },
    {
        "name": "JavaScript -> TypeScript",
        "body": {
            "source_code": 'const count = 42;\nfunction greet(name) { return name; }',
            "source_lang": "javascript",
            "target_lang": "typescript"
        }
    },
]

print("=" * 50)
print("Altern API Test")
print("=" * 50)

for test in test_cases:
    print(f"\n[{test['name']}]")
    r = requests.post("http://localhost:8000/translate", json=test["body"])
    if r.status_code == 200:
        print(r.json()["converted_code"])
    else:
        print(f"ERROR {r.status_code}: {r.text}")
    print("-" * 40)
