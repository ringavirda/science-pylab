"""Tests for the Phase 1 prose guard."""
import textwrap

import pytest

from prose_guard import check, strip_prose


def _d(s):
    return textwrap.dedent(s)


def test_docstring_rewrite_passes():
    before = _d('''
        def f(x):
            """Compute the thing -- and explain **why** at length."""
            return x + 1
    ''')
    after = _d('''
        def f(x):
            """Add one."""
            return x + 1
    ''')
    assert check(before, after)


def test_comment_rewrite_passes():
    before = _d('''
        # The old veto dropped the whole family, which is why we changed it.
        VALUE = 3
    ''')
    after = _d('''
        # Families are kept; AIC decides.
        VALUE = 3
    ''')
    assert check(before, after)


def test_docstring_removal_passes():
    before = _d('''
        def f():
            """Gone."""
            return 1
    ''')
    after = _d('''
        def f():
            return 1
    ''')
    assert check(before, after)


def test_docstring_only_body_passes():
    before = _d('''
        def f():
            """Long explanation -- with **bold**."""
    ''')
    after = _d('''
        def f():
            """Short."""
    ''')
    assert check(before, after)


def test_reflow_passes():
    before = _d('''
        def f(a, b):
            return a + b
    ''')
    after = _d('''
        def f(
            a,
            b,
        ):
            return a + b
    ''')
    assert check(before, after)


def test_logic_change_detected():
    before = "VALUE = 3\n"
    after = "VALUE = 4\n"
    assert not check(before, after)


def test_string_literal_edit_detected():
    """The failure mode a naive `--` codemod would cause."""
    before = 'SEP = " -- "\n'
    after = 'SEP = " — "\n'
    assert not check(before, after)


def test_regex_literal_edit_detected():
    before = 'PAT = re.compile(r"a--b")\n'
    after = 'PAT = re.compile(r"a—b")\n'
    assert not check(before, after)


def test_bare_string_expression_is_not_a_docstring():
    """A string in a non-docstring position is code, so edits must be caught."""
    before = _d('''
        x = 1
        "not a docstring -- just an expression"
    ''')
    after = _d('''
        x = 1
        "not a docstring, just an expression"
    ''')
    assert not check(before, after)


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        strip_prose("def f(:\n")
