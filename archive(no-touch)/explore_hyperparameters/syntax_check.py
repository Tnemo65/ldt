#!/usr/bin/env python3
import glob, ast
ok = 0
fail = []
for f in sorted(glob.glob('*.py')):
    try:
        ast.parse(open(f, encoding='utf-8').read())
        ok += 1
    except SyntaxError as e:
        fail.append(f'{f}:{e.lineno}:{e.msg}')
    except Exception as e:
        fail.append(f'{f}:UNKNOWN:{e}')
print(f'OK: {ok}/{len(glob.glob("*.py"))}')
for e in fail:
    print(f'  FAIL: {e}')
