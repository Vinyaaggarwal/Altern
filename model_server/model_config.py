"""
Model Configuration & Language Registry for Altern Local Translation AI
"""

# Base Open-Source Code Model Configuration
DEFAULT_BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
FALLBACK_BASE_MODEL = "Salesforce/codet5p-220m"

# Supported Core Programming Languages (Phase 1)
SUPPORTED_LANGUAGES = {
    "cpp": {
        "name": "C++",
        "extension": ".cpp",
        "aliases": ["c++", "cpp", "cplusplus"]
    },
    "java": {
        "name": "Java",
        "extension": ".java",
        "aliases": ["java"]
    },
    "python": {
        "name": "Python",
        "extension": ".py",
        "aliases": ["python", "py", "python3"]
    },
    "javascript": {
        "name": "JavaScript",
        "extension": ".js",
        "aliases": ["javascript", "js"]
    },
    "typescript": {
        "name": "TypeScript",
        "extension": ".ts",
        "aliases": ["typescript", "ts"]
    }
}

def normalize_language(lang_input: str) -> str:
    """Normalizes input language string to standard key."""
    clean_lang = lang_input.strip().lower()
    for key, info in SUPPORTED_LANGUAGES.items():
        if clean_lang == key or clean_lang in info["aliases"]:
            return key
    raise ValueError(f"Unsupported language: '{lang_input}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}")

def build_translation_prompt(source_code: str, source_lang: str, target_lang: str) -> str:
    """Builds a structured system prompt for code translation."""
    src_info = SUPPORTED_LANGUAGES[source_lang]
    tgt_info = SUPPORTED_LANGUAGES[target_lang]
    
    prompt = (
        f"You are an expert software engineer and compiler specialist.\n"
        f"Convert the following valid {src_info['name']} source code into equivalent, idiomatic {tgt_info['name']} code.\n"
        f"Preserve all functionality, logic, variable structures, and return types.\n\n"
        f"### Source Code ({src_info['name']}):\n"
        f"```{source_lang}\n{source_code}\n```\n\n"
        f"### Target Code ({tgt_info['name']}):\n"
        f"```{target_lang}\n"
    )
    return prompt

# Generation Parameters for LLM Inference
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "do_sample": False
}

MAX_CHUNK_LINES = 40

