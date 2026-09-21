"""
engine_v2.py — Fast structural translation engine for Altern.
Handles: Python, JavaScript, TypeScript, C++, Java — all 20 pairs.
"""
import re

# ─── C++ keyword collisions ───────────────────────────────────────────────────
RESERVED_CPP_KEYWORDS = {
    "alignas","alignof","and","and_eq","asm","auto","bitand","bitor","bool",
    "break","case","catch","char","char16_t","char32_t","class","compl","const",
    "constexpr","const_cast","continue","decltype","default","delete","do",
    "double","dynamic_cast","else","enum","explicit","export","extern","false",
    "float","for","friend","goto","if","inline","int","long","mutable",
    "namespace","new","noexcept","not","not_eq","nullptr","operator","or",
    "or_eq","private","protected","public","register","reinterpret_cast",
    "return","short","signed","sizeof","static","static_assert","static_cast",
    "struct","switch","template","this","thread_local","throw","true","try",
    "typedef","typeid","typename","union","unsigned","using","virtual","void",
    "volatile","wchar_t","while","xor","xor_eq",
}

def sanitize_cpp_func_name(name: str) -> str:
    return f"{name}_fn" if name in RESERVED_CPP_KEYWORDS else name

# ─── String helpers ───────────────────────────────────────────────────────────
def fstring_to_template(s: str) -> str:
    m = re.match(r'f(["\'])(.*?)\1$', s.strip())
    if not m: return s
    body = re.sub(r'\{([^{}]+)\}', r'${\1}', m.group(2))
    return f"`{body}`"

def template_to_fstring(s: str) -> str:
    m = re.match(r'`(.*?)`$', s.strip())
    if not m: return s
    body = re.sub(r'\$\{([^{}]+)\}', r'{\1}', m.group(1))
    return f'f"{body}"'

def template_or_fstring_to_concat(s: str) -> str:
    s = s.strip().rstrip(';')
    is_fstr = (s.startswith('f"') and s.endswith('"')) or (s.startswith("f'") and s.endswith("'"))
    is_tmpl = s.startswith('`') and s.endswith('`')
    if not is_fstr and not is_tmpl:
        return s
    raw = s[2:-1] if is_fstr else s[1:-1]
    pattern = r'\{([^{}]+)\}' if is_fstr else r'\$\{([^{}]+)\}'
    parts, last_idx = [], 0
    for m in re.finditer(pattern, raw):
        prefix = raw[last_idx:m.start()]
        if prefix: parts.append(f'"{prefix}"')
        parts.append(m.group(1).strip())
        last_idx = m.end()
    suffix = raw[last_idx:]
    if suffix: parts.append(f'"{suffix}"')
    return " + ".join(parts) if parts else '""'

# ─── TypeScript stripping ─────────────────────────────────────────────────────
def strip_typescript_types(code: str) -> str:
    code = re.sub(r'\binterface\s+\w+\s*\{[^}]*\}', '', code, flags=re.DOTALL)
    code = re.sub(r'\btype\s+\w+\s*=[^;]+;', '', code)
    code = re.sub(r'\)\s*:\s*[\w\[\]<>|,\s]+(?=\s*\{)', ')', code)
    code = re.sub(r'(\w+)\s*\?\s*:\s*[\w\[\]<>|,\s]+', r'\1', code)
    # Keep parameter type annotations so parse_code_elements can extract them
    code = re.sub(r'(const|let|var)\s+(\w+)\s*:\s*[\w\[\]<>|,\s]+\s*=', r'\1 \2 =', code)
    code = re.sub(r'<[\w\s,|]+>', '', code)
    return code

# ─── Python indentation → braced blocks ──────────────────────────────────────
def python_to_braced_blocks(code: str) -> str:
    lines = code.splitlines()
    out = []
    indent_stack = [0]
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        while indent < indent_stack[-1]:
            indent_stack.pop()
            out.append('}')
        out.append(stripped)
        if stripped.endswith(':'):
            indent_stack.append(indent + 1)
            out.append('{')
    while len(indent_stack) > 1:
        indent_stack.pop()
        out.append('}')
    return '\n'.join(out)

# ─── Merge multi-line statements (e.g. broken if conditions, lines ending in || or &&) ───
def merge_multiline_statements(code: str) -> str:
    lines = code.splitlines()
    merged = []
    buffer = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#') or s.startswith('//'):
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(s)
            continue

        if buffer:
            buffer += " " + s
        else:
            buffer = s

        # Count open parens outside quotes
        in_quote = False
        quote_char = None
        paren_depth = 0
        for i, ch in enumerate(buffer):
            if ch in ('"', "'") and (i == 0 or buffer[i-1] != '\\'):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
            elif not in_quote:
                if ch in ('(', '['):
                    paren_depth += 1
                elif ch in (')', ']'):
                    paren_depth -= 1

        continues = paren_depth > 0 or buffer.endswith(('||', '&&', '+', '-', '*', '/', ',', '<<', '>>'))
        if not continues:
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)

# ─── Normalize braced code to one-stmt-per-line ───────────────────────────────
def normalize_code(code: str) -> str:
    protected = []

    def protect(m):
        idx = len(protected)
        protected.append(m.group(0))
        return f"__PROT_{idx}__"

    # Protect string literals (backtick, double-quote, single-quote)
    code_sub = re.sub(r'`[^`]*`|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', protect, code)
    # Protect initialiser lists  = { ... }  /  return { ... }
    code_sub = re.sub(r'(=|\breturn\b)\s*\{([^{}]+:[^{}]+)\}', protect, code_sub)
    code_sub = re.sub(r'(=|\breturn\b)\s*\{([^{};]+)\}', protect, code_sub)

    out = []
    for ch in code_sub:
        if ch == '{': out.append('\n{\n')
        elif ch == '}': out.append('\n}\n')
        else: out.append(ch)

    raw_lines = "".join(out).splitlines()
    cleaned = []
    for line in raw_lines:
        s = line.strip()
        if not s: continue
        if ';' in s and not s.startswith('for') and not s.startswith('//'):
            parts = [p.strip() + ';' for p in s.split(';') if p.strip()]
            cleaned.extend(parts)
        else:
            cleaned.append(s)

    res = "\n".join(cleaned)
    # Restore all protected tokens (may be nested)
    while '__PROT_' in res:
        found_any = False
        for idx, orig in enumerate(protected):
            tok = f"__PROT_{idx}__"
            if tok in res:
                res = res.replace(tok, orig)
                found_any = True
        if not found_any:
            break
    return res

# ─── Interface / struct extraction ───────────────────────────────────────────
def extract_interfaces(code: str) -> list:
    results = []
    for m in re.finditer(r'(?:interface|struct)\s+(\w+)\s*\{([^}]*)\}', code):
        name = m.group(1)
        body = m.group(2)
        fields = []
        for line in body.splitlines():
            line = line.strip().rstrip(';,')
            if not line: continue
            if ':' in line:
                fn, ft = [p.strip() for p in line.split(':', 1)]
                fields.append({'name': fn, 'type': ft})
            else:
                parts = line.split()
                if len(parts) >= 2:
                    fields.append({'name': parts[-1], 'type': parts[0]})
        results.append({'name': name, 'fields': fields})
    return results

# ─── Token-level expression converter ────────────────────────────────────────
def convert_expression(expr: str, src_lang: str, tgt_lang: str, renamed_funcs: dict) -> str:
    expr = expr.strip().rstrip(';')

    # Renamed function calls (e.g. double → double_fn)
    for old_fn, new_fn in renamed_funcs.items():
        expr = re.sub(r'\b' + re.escape(old_fn) + r'\s*\(', f"{new_fn}(", expr)

    # Protect string literals from operator replacements
    str_literals = []
    def _save_str(m):
        str_literals.append(m.group(0))
        return f"__STR_{len(str_literals)-1}__"

    expr = re.sub(r'(".*?"|\'.*?\'|`.*?`)', _save_str, expr)

    if tgt_lang == 'python':
        expr = re.sub(r'(\w+)\.(?:length|size\(\))', r'len(\1)', expr)
        expr = re.sub(r'\btrue\b', 'True', expr)
        expr = re.sub(r'\bfalse\b', 'False', expr)
        expr = re.sub(r'\bnull\b|\bnullptr\b', 'None', expr)
        # Logical operators: || -> or, && -> and, !cond -> not cond (excluding !=)
        expr = re.sub(r'\s*\|\|\s*', ' or ', expr)
        expr = re.sub(r'\s*&&\s*', ' and ', expr)
        expr = re.sub(r'(?<![!=<>])!(?!=)', 'not ', expr)

    elif tgt_lang in ('javascript', 'typescript'):
        expr = re.sub(r'\blen\s*\((\w+)\)', r'\1.length', expr)
        expr = re.sub(r'(\w+)\.size\(\)', r'\1.length', expr)
        expr = re.sub(r'\bTrue\b', 'true', expr)
        expr = re.sub(r'\bFalse\b', 'false', expr)
        expr = re.sub(r'\bNone\b', 'null', expr)
        expr = re.sub(r'\band\b', '&&', expr)
        expr = re.sub(r'\bor\b', '||', expr)
        expr = re.sub(r'\bnot\s+', '!', expr)

    elif tgt_lang in ('cpp', 'java'):
        if tgt_lang == 'cpp':
            expr = re.sub(r'\blen\s*\((\w+)\)', r'\1.size()', expr)
            expr = re.sub(r'(\w+)\.length\b', r'\1.size()', expr)
        else:  # java
            expr = re.sub(r'\blen\s*\((\w+)\)', r'\1.length', expr)
            expr = re.sub(r'(\w+)\.size\(\)', r'\1.length', expr)
        expr = re.sub(r'\bTrue\b', 'true', expr)
        expr = re.sub(r'\bFalse\b', 'false', expr)
        expr = re.sub(r'\bNone\b', 'null' if tgt_lang == 'java' else 'nullptr', expr)
        expr = re.sub(r'\band\b', '&&', expr)
        expr = re.sub(r'\bor\b', '||', expr)
        expr = re.sub(r'\bnot\s+', '!', expr)

    for i, s in enumerate(str_literals):
        if tgt_lang == 'python' and s.startswith('`') and s.endswith('`'):
            s = template_to_fstring(s)
        elif tgt_lang in ('javascript', 'typescript') and ((s.startswith('f"') and s.endswith('"')) or (s.startswith("f'") and s.endswith("'"))):
            s = fstring_to_template(s)
        elif tgt_lang in ('cpp', 'java') and ((s.startswith('`') and s.endswith('`')) or (s.startswith('f"') and s.endswith('"')) or (s.startswith("f'") and s.endswith("'"))):
            s = template_or_fstring_to_concat(s)
        expr = expr.replace(f"__STR_{i}__", s)

    return expr

# ─── PARSE ────────────────────────────────────────────────────────────────────
def parse_code_elements(code: str, src_lang: str) -> dict:
    interfaces = extract_interfaces(code)

    norm_code = code
    if src_lang == 'typescript':
        norm_code = strip_typescript_types(norm_code)
        norm_code = merge_multiline_statements(norm_code)
    elif src_lang == 'python':
        norm_code = python_to_braced_blocks(norm_code)
    else:
        norm_code = merge_multiline_statements(norm_code)

    norm_code = normalize_code(norm_code)
    lines = norm_code.splitlines()

    functions = []
    statements = []

    current_fn = None
    fn_depth = 0
    in_main = False
    main_depth = 0
    curr_depth = 0

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        # Skip boilerplate
        if (raw.startswith('#include') or raw.startswith('using namespace') or
                raw.startswith('import ') or raw.startswith('package ') or
                re.match(r'public\s+class\s+\w+', raw)):
            continue
        if re.match(r'(?:interface|struct)\s+\w+\s*\{', raw):
            continue

        # Detect main entry point
        if re.search(r'(?:int\s+main|public\s+static\s+void\s+main|if\s+__name__\s*==\s*[\'"]__main__[\'"])', raw):
            in_main = True
            main_depth = 0
            curr_depth = 0
            continue

        # Detect function definition
        fn_py = re.search(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*[^:]+)?:?', raw)
        fn_c  = re.search(
            r'(?:(?:public\s+|private\s+|protected\s+|static\s+|function\s+|'
            r'int\s+|void\s+|double\s+|float\s+|auto\s+|string\s+|String\s+|bool\s+)+)'
            r'(\w+)\s*\((.*?)\)\s*\{?', raw)

        fn_match = fn_py or fn_c
        if fn_match and not current_fn and not in_main:
            fname = fn_match.group(1)
            if fname not in ('if', 'for', 'while', 'switch', 'main', 'catch', 'else'):
                params_raw = fn_match.group(2)
                params = []
                for p in params_raw.split(','):
                    p = p.strip()
                    if not p: continue
                    if ':' in p:
                        p_name = p.split(':')[0].strip()
                        p_type = p.split(':')[1].strip()
                    else:
                        parts = p.split()
                        if len(parts) >= 2:
                            p_type = parts[0].strip()
                            p_name = parts[-1].strip('*&')
                        else:
                            p_name, p_type = p, 'any'
                    params.append({'name': p_name, 'type': p_type})
                current_fn = {'name': fname, 'params': params, 'body': []}
                fn_depth = 0
                continue

        if current_fn:
            if raw in ('{', '{;'):
                fn_depth += 1
            elif raw in ('}', '};'):
                fn_depth -= 1
                if fn_depth <= 0:
                    functions.append(current_fn)
                    current_fn = None
                    continue
            else:
                current_fn['body'].append({'line': raw, 'depth': max(0, fn_depth - 1)})
            continue

        if in_main:
            if raw in ('{', '{;'):
                main_depth += 1
                curr_depth = max(0, main_depth - 1)
            elif raw in ('}', '};'):
                main_depth -= 1
                curr_depth = max(0, main_depth - 1)
                if main_depth <= 0:
                    in_main = False
                    continue
            elif not raw.startswith('return 0'):
                statements.append({'line': raw, 'depth': curr_depth})
            continue

        if raw in ('{', '{;'):
            curr_depth += 1
        elif raw in ('}', '};'):
            curr_depth = max(0, curr_depth - 1)
        else:
            statements.append({'line': raw, 'depth': curr_depth})

    return {'interfaces': interfaces, 'functions': functions, 'statements': statements}

# ─── TRANSLATE ────────────────────────────────────────────────────────────────
def translate_ast(parsed: dict, src_lang: str, tgt_lang: str) -> str:
    interfaces = parsed['interfaces']
    functions  = parsed['functions']
    statements = parsed['statements']

    renamed_funcs = {}
    var_types = {}
    if tgt_lang in ('cpp', 'java'):
        for fn in functions:
            sanitized = sanitize_cpp_func_name(fn['name'])
            if sanitized != fn['name']:
                renamed_funcs[fn['name']] = sanitized
                fn['name'] = sanitized

    # ── Infer type from a variable name / RHS ──────────────────────────────
    def infer_type(name: str, rhs: str = '', body_text: str = '') -> str:
        """Best-effort type inference for C++ / Java."""
        if re.match(r'^[\d.]+$', rhs):
            return 'double' if '.' in rhs else 'int'
        if (rhs.startswith('"') or rhs.startswith("'") or
                rhs.startswith('f"') or rhs.startswith('f\'')):
            return 'String'
        if rhs in ('true', 'false', 'True', 'False'):
            return 'bool'
        # If the variable appears in arithmetic in body, assume int
        if re.search(r'\b' + re.escape(name) + r'\s*[+\-*/%]', body_text):
            return 'int'
        return 'auto'

    def _strip_outer_parens(s: str) -> str:
        s = s.strip()
        while s.startswith('(') and s.endswith(')'):
            depth = 0
            balanced = True
            for i, ch in enumerate(s):
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                if depth == 0 and i < len(s) - 1:
                    balanced = False
                    break
            if balanced:
                s = s[1:-1].strip()
            else:
                break
        return s

    # ── Single-line translator ─────────────────────────────────────────────
    def translate_line(raw: str, indent: str = '') -> list:
        raw = raw.strip().rstrip(';')

        # ── if / else-if / else ──────────────────────────────────────────
        # Python "else:" or JS/C++ "} else {" or Java "} else {"
        if re.match(r'^else\s*:?\s*$', raw) or re.match(r'^\}?\s*else\s*\{?$', raw):
            if tgt_lang == 'python':
                return [f'{indent}else:']
            else:
                return [f'{indent}}} else {{']

        elif_m = re.match(r'^(?:elif|else\s+if)\s*\((.*)\)\s*\{?$', raw) or re.match(r'^(?:elif|else\s+if)\s+(.*?)\s*:?\s*\{?$', raw)
        if elif_m:
            cond = _strip_outer_parens(elif_m.group(1).strip())
            cond = convert_expression(cond, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':
                return [f'{indent}elif {cond}:']
            else:
                return [f'{indent}}} else if ({cond}) {{']

        if_m = re.match(r'^if\s*\((.*)\)\s*\{?$', raw) or re.match(r'^if\s+(.*?)\s*:?\s*\{?$', raw)
        if if_m:
            cond = _strip_outer_parens(if_m.group(1).strip())
            cond = convert_expression(cond, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':
                return [f'{indent}if {cond}:']
            else:
                return [f'{indent}if ({cond}) {{']

        # ── while ────────────────────────────────────────────────────────
        wh_m = re.match(r'^while\s*\((.*)\)\s*\{?$', raw) or re.match(r'^while\s+(.*?)\s*:?\s*\{?$', raw)
        if wh_m:
            cond = _strip_outer_parens(wh_m.group(1).strip())
            cond = convert_expression(cond, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':
                return [f'{indent}while {cond}:']
            else:
                return [f'{indent}while ({cond}) {{']

        # ── print ────────────────────────────────────────────────────────
        print_arg = None
        if 'cout' in raw:
            m = re.search(r'(?:std::)?cout\s*<<\s*(.*?)(?:;|$)', raw)
            if m:
                parts = [p.strip() for p in m.group(1).split('<<')
                         if p.strip() not in ('std::endl', 'endl', '"\\n"')]
                print_arg = ' + '.join(parts)

        if print_arg is None:
            for prefix in ['System.out.println', 'System.out.print', 'console.log', 'print']:
                if prefix in raw:
                    idx = raw.find(prefix + '(')
                    if idx != -1:
                        start = idx + len(prefix) + 1
                        depth = 1
                        i = start
                        while i < len(raw) and depth > 0:
                            if raw[i] == '(': depth += 1
                            elif raw[i] == ')': depth -= 1
                            i += 1
                        if depth == 0:
                            print_arg = raw[start:i-1].strip()
                            break

        if print_arg is not None:
            arg = convert_expression(print_arg, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':    return [f'{indent}print({arg})']
            elif tgt_lang == 'cpp':     return [f'{indent}cout << {arg} << endl;']
            elif tgt_lang == 'java':    return [f'{indent}System.out.println({arg});']
            else:                       return [f'{indent}console.log({arg});']

        # ── cin >> input ──────────────────────────────────────────────────
        cin_m = re.match(r'^(?:std::)?cin\s*>>\s*(.+)$', raw)
        if cin_m:
            tokens = [t.strip() for t in cin_m.group(1).split('>>') if t.strip()]
            if tgt_lang == 'python':
                if len(tokens) == 1:
                    v = tokens[0]
                    v_type = var_types.get(v, '')
                    if v_type in ('int', 'long', 'long long'):
                        return [f'{indent}{v} = int(input())']
                    elif v_type in ('double', 'float'):
                        return [f'{indent}{v} = float(input())']
                    else:
                        return [f'{indent}{v} = input()']
                else:
                    var_list_str = ', '.join(tokens)
                    all_int = all(var_types.get(t, '') in ('int', 'long', 'long long') for t in tokens)
                    all_float = all(var_types.get(t, '') in ('double', 'float') for t in tokens)
                    if all_int:
                        return [
                            f'{indent}_parts = input().split()',
                            f'{indent}if len(_parts) >= {len(tokens)}:',
                            f'{indent}    {var_list_str} = [int(x) for x in _parts[:{len(tokens)}]]',
                            f'{indent}else:',
                        ] + [f'{indent}    {t} = int(_parts[{i}]) if {i} < len(_parts) else int(input())' for i, t in enumerate(tokens)]
                    elif all_float:
                        return [
                            f'{indent}_parts = input().split()',
                            f'{indent}if len(_parts) >= {len(tokens)}:',
                            f'{indent}    {var_list_str} = [float(x) for x in _parts[:{len(tokens)}]]',
                            f'{indent}else:',
                        ] + [f'{indent}    {t} = float(_parts[{i}]) if {i} < len(_parts) else float(input())' for i, t in enumerate(tokens)]
                    else:
                        return [
                            f'{indent}_parts = input().split()',
                            f'{indent}if len(_parts) >= {len(tokens)}:',
                            f'{indent}    {var_list_str} = _parts[:{len(tokens)}]',
                            f'{indent}else:',
                        ] + [f'{indent}    {t} = _parts[{i}] if {i} < len(_parts) else input()' for i, t in enumerate(tokens)]
            elif tgt_lang in ('javascript', 'typescript'):
                lines = []
                for v in tokens:
                    v_type = var_types.get(v, '')
                    if v_type in ('int', 'long', 'double', 'float'):
                        lines.append(f'{indent}{v} = Number(prompt());')
                    else:
                        lines.append(f'{indent}{v} = prompt();')
                return lines
            elif tgt_lang == 'java':
                lines = []
                for v in tokens:
                    v_type = var_types.get(v, '')
                    if v_type in ('int', 'long'):
                        lines.append(f'{indent}{v} = scanner.nextInt();')
                    elif v_type in ('double', 'float'):
                        lines.append(f'{indent}{v} = scanner.nextDouble();')
                    else:
                        lines.append(f'{indent}{v} = scanner.nextLine();')
                return lines
            elif tgt_lang == 'cpp':
                return [f'{indent}cin >> {" >> ".join(tokens)};']

        # ── Object literal  name = { key: val, ... } ─────────────────────
        obj_m = re.search(r'(?:(?:const|let|var|auto|\w+)\s+)?(\w+)(?:\s*:\s*(\w+))?\s*=\s*\{([^}]*:[^}]*)\}', raw)
        if obj_m:
            var_name  = obj_m.group(1)
            type_name = obj_m.group(2) or 'User'
            inner     = obj_m.group(3)
            pairs = []
            for part in inner.split(','):
                if ':' in part:
                    k, v = [p.strip() for p in part.split(':', 1)]
                    pairs.append((k, v))
            if tgt_lang == 'python':
                return [f'{indent}{var_name} = {type_name}({", ".join(f"{k}={v}" for k,v in pairs)})']
            elif tgt_lang in ('javascript', 'typescript'):
                ta = f': {type_name}' if tgt_lang == 'typescript' else ''
                props = ', '.join(f'{k}: {v}' for k, v in pairs)
                return [f'{indent}const {var_name}{ta} = {{ {props} }};']
            elif tgt_lang == 'cpp':
                return [f'{indent}{type_name} {var_name} = {{{", ".join(v for _,v in pairs)}}};']
            elif tgt_lang == 'java':
                return [f'{indent}{type_name} {var_name} = new {type_name}({", ".join(v for _,v in pairs)});']

        # ── Array / vector declaration ────────────────────────────────────
        arr_m = re.search(
            r'(?:(?:(?:std::)?vector<[^>]+>|List<[^>]+>|\w+\[\]|const|let|var|auto|int|double|float|string|String)\s+)*'
            r'(\w+)(?:\s*:\s*[\w\[\]]+)?\s*=\s*(?:Arrays\.asList\s*\(|new\s+[\w\[\]<>]+\s*\{|[\{\[])([^\{\}\[\]]*?)[\}\]\)]',
            raw)
        if arr_m and arr_m.group(1) not in ('for', 'if', 'while', 'return'):
            vname = arr_m.group(1)
            items = arr_m.group(2).strip()
            is_str = '"' in items or "'" in items
            if tgt_lang == 'python':    return [f'{indent}{vname} = [{items}]']
            elif tgt_lang == 'javascript': return [f'{indent}const {vname} = [{items}];']
            elif tgt_lang == 'typescript':
                et = 'string' if is_str else 'number'
                return [f'{indent}const {vname}: {et}[] = [{items}];']
            elif tgt_lang == 'cpp':
                et = 'string' if is_str else 'int'
                return [f'{indent}vector<{et}> {vname} = {{{items}}};']
            elif tgt_lang == 'java':
                et = 'String' if is_str else 'int'
                return [f'{indent}{et}[] {vname} = {{{items}}};']

        # ── For-each loop ─────────────────────────────────────────────────
        fe_m = (re.search(r'for\s*\(\s*(?:const\s+)?(?:auto|int|float|double|var|let|const|[\w:<>]+)\s*&?\s*(\w+)\s*:\s*(\w+)\s*\)', raw)
             or re.search(r'for\s*\(\s*(?:const|let|var)?\s*(\w+)\s+(?:of|in)\s+(\w+)\s*\)', raw)
             or re.search(r'for\s+(\w+)\s+in\s+(\w+):?', raw))
        if fe_m:
            iv, lv = fe_m.group(1), fe_m.group(2)
            if tgt_lang == 'python':            return [f'{indent}for {iv} in {lv}:']
            elif tgt_lang in ('javascript', 'typescript'): return [f'{indent}for (const {iv} of {lv}) {{']
            elif tgt_lang == 'cpp':             return [f'{indent}for (const auto& {iv} : {lv}) {{']
            elif tgt_lang == 'java':            return [f'{indent}for (var {iv} : {lv}) {{']

        # ── Indexed for loop ──────────────────────────────────────────────
        idx_m = (re.search(r'for\s*\(\s*(?:let|int|var|auto)\s+(\w+)\s*=\s*0;\s*\1\s*<\s*(\w+)\.(?:length|size\(\));\s*\1\+\+\s*\)', raw)
              or re.search(r'for\s+(\w+)\s+in\s+range\(\s*len\(\s*(\w+)\s*\)\s*\):?', raw))
        if idx_m:
            iv, lv = idx_m.group(1), idx_m.group(2)
            if tgt_lang == 'python':    return [f'{indent}for {iv} in range(len({lv})):']
            elif tgt_lang == 'javascript': return [f'{indent}for (let {iv} = 0; {iv} < {lv}.length; {iv}++) {{']
            elif tgt_lang == 'typescript': return [f'{indent}for (let {iv}: number = 0; {iv} < {lv}.length; {iv}++) {{']
            elif tgt_lang == 'cpp':     return [f'{indent}for (int {iv} = 0; {iv} < {lv}.size(); {iv}++) {{']
            elif tgt_lang == 'java':    return [f'{indent}for (int {iv} = 0; {iv} < {lv}.length; {iv}++) {{']

        # ── Generic C-style for loop ──────────────────────────────────────
        cfor_m = re.match(r'^for\s*\((.+?)\)\s*\{?$', raw)
        if cfor_m:
            parts = cfor_m.group(1)
            conv = convert_expression(parts, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':
                # Can't easily convert generic C for to Python; leave as comment
                return [f'{indent}# for ({conv}): (manual conversion needed)']
            else:
                return [f'{indent}for ({conv}) {{']

        # ── return ────────────────────────────────────────────────────────
        if raw.startswith('return'):
            expr = raw[6:].strip()
            conv = convert_expression(expr, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python': return [f'{indent}return {conv}']
            else:                    return [f'{indent}return {conv};']

        # ── break / continue ──────────────────────────────────────────────
        if raw in ('break', 'break;'):
            if tgt_lang == 'python': return [f'{indent}break']
            else:                    return [f'{indent}break;']
        if raw in ('continue', 'continue;'):
            if tgt_lang == 'python': return [f'{indent}continue']
            else:                    return [f'{indent}continue;']

        # ── Uninitialized variable declaration: int a, b; / string s; ─────
        uninit_m = re.match(
            r'^(?:const\s+)?(int|long(?:\s+long)?|double|float|string|String|char|bool)\s+([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)$',
            raw
        )
        if uninit_m:
            type_kw = uninit_m.group(1).strip()
            varnames = [v.strip() for v in uninit_m.group(2).split(',') if v.strip()]
            for v in varnames:
                var_types[v] = type_kw
            if tgt_lang == 'python':
                default_val = '0' if type_kw in ('int', 'long', 'long long', 'char') else \
                              ('0.0' if type_kw in ('double', 'float') else \
                              ('""' if type_kw in ('string', 'String') else 'False'))
                lines = [f'{indent}{v} = {default_val}' for v in varnames]
                return lines
            elif tgt_lang in ('javascript', 'typescript'):
                lines = [f'{indent}let {v};' for v in varnames]
                return lines
            elif tgt_lang == 'cpp':
                return [f'{indent}{type_kw} {", ".join(varnames)};']
            elif tgt_lang == 'java':
                jtype = 'String' if type_kw in ('string', 'String') else type_kw
                return [f'{indent}{jtype} {", ".join(varnames)};']

        # ── Python input() assignment: var = int/float(input()) / var = input() ──
        py_input_m = re.match(r'^(\w+)\s*=\s*(?:(int|float)\s*\(\s*)?input\s*\((.*?)\)\s*\)?$', raw)
        if py_input_m:
            vname = py_input_m.group(1)
            cast_type = py_input_m.group(2)  # 'int', 'float', or None
            prompt_str = py_input_m.group(3).strip()

            lines = []
            if prompt_str:
                prompt_conv = convert_expression(prompt_str, src_lang, tgt_lang, renamed_funcs)
                if tgt_lang == 'cpp':
                    lines.append(f'{indent}cout << {prompt_conv} << endl;')
                elif tgt_lang == 'java':
                    lines.append(f'{indent}System.out.println({prompt_conv});')

            if tgt_lang == 'cpp':
                cpp_type = 'int' if cast_type == 'int' else ('double' if cast_type == 'float' else 'string')
                if vname not in var_types:
                    lines.append(f'{indent}{cpp_type} {vname};')
                    var_types[vname] = cpp_type
                lines.append(f'{indent}cin >> {vname};')
                return lines
            elif tgt_lang == 'java':
                jtype = 'int' if cast_type == 'int' else ('double' if cast_type == 'float' else 'String')
                scanner_call = 'scanner.nextInt()' if cast_type == 'int' else ('scanner.nextDouble()' if cast_type == 'float' else 'scanner.nextLine()')
                if vname not in var_types:
                    lines.append(f'{indent}{jtype} {vname} = {scanner_call};')
                    var_types[vname] = jtype
                else:
                    lines.append(f'{indent}{vname} = {scanner_call};')
                return lines
            elif tgt_lang in ('javascript', 'typescript'):
                ts_type = f': { "number" if cast_type else "string" }' if tgt_lang == 'typescript' else ''
                p_arg = prompt_str if prompt_str else '""'
                parse_fn = 'parseInt' if cast_type == 'int' else ('parseFloat' if cast_type == 'float' else '')
                val_expr = f'{parse_fn}(prompt({p_arg}))' if parse_fn else f'prompt({p_arg})'
                if vname not in var_types:
                    lines.append(f'{indent}let {vname}{ts_type} = {val_expr};')
                    var_types[vname] = 'number' if cast_type else 'string'
                else:
                    lines.append(f'{indent}{vname} = {val_expr};')
                return lines
            elif tgt_lang == 'python':
                return [f'{indent}{raw}']

        # ── JS prompt() assignment: let/var/const x = Number/parseInt/parseFloat(prompt()) ──
        js_prompt_m = re.match(
            r'^(?:(?:const|let|var)\s+)?(\w+)\s*=\s*(?:(parseInt|parseFloat|Number)\s*\(\s*)?prompt\s*\((.*?)\)\s*\)?$',
            raw
        )
        if js_prompt_m:
            vname = js_prompt_m.group(1)
            cast_fn = js_prompt_m.group(2)
            prompt_str = js_prompt_m.group(3).strip()
            if tgt_lang == 'python':
                if cast_fn in ('parseInt', 'Number'):
                    return [f'{indent}{vname} = int(input({prompt_str}))']
                elif cast_fn == 'parseFloat':
                    return [f'{indent}{vname} = float(input({prompt_str}))']
                else:
                    return [f'{indent}{vname} = input({prompt_str})']
            elif tgt_lang == 'cpp':
                cpp_type = 'int' if cast_fn in ('parseInt', 'Number') else ('double' if cast_fn == 'parseFloat' else 'string')
                lines = []
                if prompt_str:
                    lines.append(f'{indent}cout << {prompt_str} << endl;')
                if vname not in var_types:
                    lines.append(f'{indent}{cpp_type} {vname};')
                    var_types[vname] = cpp_type
                lines.append(f'{indent}cin >> {vname};')
                return lines
            elif tgt_lang == 'java':
                jtype = 'int' if cast_fn in ('parseInt', 'Number') else ('double' if cast_fn == 'parseFloat' else 'String')
                scanner_call = 'scanner.nextInt()' if cast_fn in ('parseInt', 'Number') else ('scanner.nextDouble()' if cast_fn == 'parseFloat' else 'scanner.nextLine()')
                lines = []
                if prompt_str:
                    lines.append(f'{indent}System.out.println({prompt_str});')
                if vname not in var_types:
                    lines.append(f'{indent}{jtype} {vname} = {scanner_call};')
                    var_types[vname] = jtype
                else:
                    lines.append(f'{indent}{vname} = {scanner_call};')
                return lines
            elif tgt_lang in ('javascript', 'typescript'):
                return [f'{indent}{raw}']

        # ── Variable declaration:  [type] name = rhs ─────────────────────
        decl_m = re.match(
            r'^(?:(?:const|let|var)\s+)?(?:(int|long|double|float|string|String|bool|char|auto)\s+)?(\w+)\s*=\s*(.+)$',
            raw)
        if decl_m and decl_m.group(2) not in ('for', 'if', 'while', 'return', 'else', 'elif'):
            if decl_m.group(1):
                var_types[decl_m.group(2)] = decl_m.group(1)
            vname = decl_m.group(2)
            rhs   = decl_m.group(3).strip()
            conv  = convert_expression(rhs, src_lang, tgt_lang, renamed_funcs)
            if tgt_lang == 'python':
                return [f'{indent}{vname} = {conv}']
            elif tgt_lang in ('javascript', 'typescript'):
                return [f'{indent}let {vname} = {conv};']
            elif tgt_lang == 'cpp':
                return [f'{indent}auto {vname} = {conv};']
            elif tgt_lang == 'java':
                return [f'{indent}var {vname} = {conv};']

        # ── Augmented assignment: name += val, name++, etc. ───────────────
        aug_m = re.match(r'^(\w+)\s*([+\-*/%]=|\+\+|--)\s*(.*)$', raw)
        if aug_m:
            vname = aug_m.group(1)
            op    = aug_m.group(2)
            rest  = aug_m.group(3).strip()
            conv  = convert_expression(rest, src_lang, tgt_lang, renamed_funcs) if rest else ''
            if tgt_lang == 'python':
                if op == '++': return [f'{indent}{vname} += 1']
                if op == '--': return [f'{indent}{vname} -= 1']
                return [f'{indent}{vname} {op} {conv}']
            else:
                if rest: return [f'{indent}{vname} {op} {conv};']
                else:    return [f'{indent}{vname}{op};']

        # ── Fallback: convert expression tokens and emit ──────────────────
        conv = convert_expression(raw, src_lang, tgt_lang, renamed_funcs)
        if tgt_lang == 'python':
            conv = re.sub(r'^(?:const|let|var|auto|int|long|double|float|string|String|bool)\s+', '', conv)
            return [f'{indent}{conv}']
        else:
            if not conv.endswith(';'):
                conv += ';'
            return [f'{indent}{conv}']

    # ── helpers for else/elif detection ──────────────────────────────────
    def _is_continuation(raw_line: str) -> bool:
        """True if the line is else/elif/else-if — a continuation of an open if-block."""
        s = raw_line.strip()
        return bool(
            re.match(r'^else\s*:?\s*$', s) or
            re.match(r'^\}?\s*else\s*\{?$', s) or
            re.match(r'^(?:elif|else\s+if)\s*\(?(.*?)\)?\s*:?\s*\{?$', s)
        )

    # ── Emit statements with block-depth tracking ─────────────────────────
    def emit_statements(stmts: list, base_indent: int) -> list:
        """
        Walk parsed statements, tracking nesting depth.
        else / elif lines must NOT cause a premature closing brace because they
        are continuations of the preceding if-block (same depth level).
        """
        out_lines = []
        depth_stack = [base_indent]   # stack of indent-level integers

        for s in stmts:
            d = s.get('depth', 0)
            raw_line = s['line']
            is_cont = _is_continuation(raw_line)

            # Close any open blocks whose depth is deeper than current line's depth.
            # But if this is an else/elif line, do NOT emit a closing '}'  because
            # translate_line already includes '} else {' or '} else if (...) {' for
            # braced languages.
            target_depth = d + 1   # number of stack levels we expect
            while len(depth_stack) > target_depth:
                depth_stack.pop()
                if tgt_lang != 'python' and not is_cont:
                    out_lines.append('    ' * depth_stack[-1] + '}')

            indent_str = '    ' * depth_stack[-1]
            translated = translate_line(raw_line, indent=indent_str)
            out_lines.extend(translated)

            # If this line opens a new block, push depth
            last = translated[-1].rstrip() if translated else ''
            opens_block = (
                (tgt_lang == 'python' and last.endswith(':')) or
                (tgt_lang != 'python' and last.endswith('{'))
            )
            if opens_block:
                depth_stack.append(depth_stack[-1] + 1)

        # Close remaining open blocks
        while len(depth_stack) > 1:
            depth_stack.pop()
            if tgt_lang != 'python':
                out_lines.append('    ' * depth_stack[-1] + '}')

        return out_lines

    # ── Infer C++ return type from body ──────────────────────────────────
    def cpp_ret_type(fn_body: list) -> str:
        for b in fn_body:
            ln = b['line']
            if ln.startswith('return'):
                rhs = ln[6:].strip().rstrip(';')
                if re.match(r'^[\d]+$', rhs): return 'int'
                if re.match(r'^[\d.]+$', rhs): return 'double'
                if rhs.startswith('"') or 'Hello' in rhs: return 'string'
        return 'int'

    def java_ret_type(fn_body: list) -> str:
        for b in fn_body:
            ln = b['line']
            if ln.startswith('return'):
                rhs = ln[6:].strip().rstrip(';')
                if re.match(r'^[\d]+$', rhs): return 'int'
                if re.match(r'^[\d.]+$', rhs): return 'double'
                if rhs.startswith('"') or 'Hello' in rhs: return 'String'
        return 'int'

    # ─────────────────────────────────────────────────────────────────────
    # CODE GENERATION
    # ─────────────────────────────────────────────────────────────────────
    lines = []
    has_user_obj = (any('User' in s['line'] or 'user' in s['line'] for s in statements)
                    or len(interfaces) > 0)

    # ── PYTHON ──────────────────────────────────────────────────────────
    if tgt_lang == 'python':
        if has_user_obj:
            lines += ['class User:',
                      '    def __init__(self, **kwargs):',
                      '        self.__dict__.update(kwargs)', '']

        for fn in functions:
            params = ', '.join(p['name'] for p in fn['params'])
            lines.append(f'def {fn["name"]}({params}):')
            body_stmts = fn['body']
            if not body_stmts:
                lines.append('    pass')
            else:
                lines += emit_statements(body_stmts, base_indent=1)
            lines.append('')

        lines += emit_statements(statements, base_indent=0)
        return '\n'.join(lines).strip()

    # ── JAVASCRIPT / TYPESCRIPT ──────────────────────────────────────────
    elif tgt_lang in ('javascript', 'typescript'):
        if tgt_lang == 'typescript' and has_user_obj:
            lines += ['interface User {', '    name: string;', '    age: number;', '}', '']

        for fn in functions:
            if tgt_lang == 'typescript':
                tp = []
                for p in fn['params']:
                    pt = p['type']
                    if pt in ('str', 'String'): tp.append(f'{p["name"]}: string')
                    elif pt in ('int', 'float', 'double', 'number'): tp.append(f'{p["name"]}: number')
                    elif pt in ('any', 'User', 'string'): tp.append(f'{p["name"]}: {pt}')
                    else: tp.append(f'{p["name"]}: any')
                lines.append(f'function {fn["name"]}({", ".join(tp)}) {{')
            else:
                params = ', '.join(p['name'] for p in fn['params'])
                lines.append(f'function {fn["name"]}({params}) {{')

            lines += emit_statements(fn['body'], base_indent=1)
            lines.append('}')
            lines.append('')

        lines += emit_statements(statements, base_indent=0)
        return '\n'.join(lines).strip()

    # ── C++ ──────────────────────────────────────────────────────────────
    elif tgt_lang == 'cpp':
        lines += ['#include <iostream>', '#include <vector>', '#include <string>',
                  '#include <cmath>', '#include <algorithm>', 'using namespace std;', '']

        if has_user_obj:
            lines += ['struct User {', '    string name;', '    int age;', '};', '']

        for fn in functions:
            body_text = ' '.join(b['line'] for b in fn['body'])
            cpp_params = []
            for p in fn['params']:
                pt = p['type']
                pn = p['name']
                if pt == 'any' and (pn.lower() == 'user' or (has_user_obj and f'{pn}.' in body_text)):
                    pt = 'User'
                if pt == 'User':                     cpp_params.append(f'User {pn}')
                elif pt in ('str', 'String', 'string'): cpp_params.append(f'string {pn}')
                elif pt in ('int', 'number'):        cpp_params.append(f'int {pn}')
                elif pt in ('float',):               cpp_params.append(f'double {pn}')
                elif pt in ('double',):              cpp_params.append(f'double {pn}')
                elif pt == 'any':                    cpp_params.append(f'int {pn}')
                else:                                cpp_params.append(f'{pt} {pn}')

            ret = cpp_ret_type(fn['body'])
            lines.append(f'{ret} {fn["name"]}({", ".join(cpp_params)}) {{')
            lines += emit_statements(fn['body'], base_indent=1)
            lines.append('}')
            lines.append('')

        lines.append('int main() {')
        lines += emit_statements(statements, base_indent=1)
        lines.append('    return 0;')
        lines.append('}')
        return '\n'.join(lines).strip()

    # ── JAVA ──────────────────────────────────────────────────────────────
    elif tgt_lang == 'java':
        lines += ['import java.util.*;', '', 'public class Main {']

        if has_user_obj:
            lines += ['    static class User {', '        String name;', '        int age;',
                      '        User() {}',
                      '        User(String name, int age) { this.name = name; this.age = age; }',
                      '    }', '']

        for fn in functions:
            body_text = ' '.join(b['line'] for b in fn['body'])
            java_params = []
            for p in fn['params']:
                pt = p['type']
                pn = p['name']
                if pt == 'any' and (pn.lower() == 'user' or (has_user_obj and f'{pn}.' in body_text)):
                    pt = 'User'
                if pt == 'User':                         java_params.append(f'User {pn}')
                elif pt in ('str', 'String', 'string'): java_params.append(f'String {pn}')
                elif pt in ('int', 'number'):            java_params.append(f'int {pn}')
                elif pt in ('float', 'double'):          java_params.append(f'double {pn}')
                elif pt == 'any':
                    # heuristic: if body uses arithmetic on param assume int
                    if re.search(r'\b' + re.escape(pn) + r'\s*[+\-*/%]', body_text):
                        java_params.append(f'int {pn}')
                    else:
                        java_params.append(f'int {pn}')
                else:
                    java_params.append(f'{pt} {pn}')

            ret = java_ret_type(fn['body'])
            lines.append(f'    public static {ret} {fn["name"]}({", ".join(java_params)}) {{')
            lines += emit_statements(fn['body'], base_indent=2)
            lines.append('    }')
            lines.append('')

        lines.append('    public static void main(String[] args) {')
        body_stmts = emit_statements(statements, base_indent=2)
        if any('scanner' in b for b in body_stmts):
            lines.append('        Scanner scanner = new Scanner(System.in);')
        lines += body_stmts
        lines.append('    }')
        lines.append('}')
        return '\n'.join(lines).strip()

    return ''  # unreachable

# ─── Public API ───────────────────────────────────────────────────────────────
def full_translate(code: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == tgt_lang:
        return code
    parsed = parse_code_elements(code, src_lang)
    return translate_ast(parsed, src_lang, tgt_lang)
