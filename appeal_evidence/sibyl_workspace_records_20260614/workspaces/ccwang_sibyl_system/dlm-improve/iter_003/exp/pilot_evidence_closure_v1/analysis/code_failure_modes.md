# MBPP Failure Modes

本文件统计 MBPP 上各 arm 的失败桶，以及关键 harm 对比。

## DNB-64
- `AssertionError`: 2
- `IndentationError`: 1
- `NameError`: 23
- `SyntaxError`: 21
- `TypeError`: 1

## DNB-84
- `AssertionError`: 2
- `NameError`: 26
- `SyntaxError`: 19
- `TypeError`: 1
- `other`: 1

## CARD-84
- `AssertionError`: 1
- `IndentationError`: 4
- `NameError`: 29
- `SyntaxError`: 13
- `TypeError`: 1

## RAND-84
- `AssertionError`: 2
- `IndentationError`: 3
- `NameError`: 22
- `SyntaxError`: 20
- `TypeError`: 1

## Harm Delta
- `card84_vs_dnb64 harmed`: 0
- `dnb84_vs_dnb64 harmed`: 1
- `delta(card - dnb84)`: -1
