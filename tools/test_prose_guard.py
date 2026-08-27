"""Tests for the Phase 1 prose guard."""
import subprocess
import textwrap
from pathlib import Path

import pytest

import prose_guard
from prose_guard import _git_show, _verify_ref, check, main, strip_prose


def _d(s):
    return textwrap.dedent(s)


def _git(*args, cwd):
    """Run a git command in ``cwd``, raising on failure."""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True,
        capture_output=True, text=True, encoding="utf-8",
    )


def _commit(repo, name, content, message):
    """Write ``name`` with ``content`` and commit it in ``repo``."""
    (repo / name).write_text(content, encoding="utf-8")
    _git("add", name, cwd=repo)
    _git("commit", "-q", "-m", message, cwd=repo)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo, isolated from the real one, holding one
    committed file with real, checkable logic. cwd is switched into
    it, since prose_guard shells out to git relative to cwd.
    """
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    _commit(tmp_path, "a.py", "VALUE = 1\n", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


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


def test_docstring_vs_pass_collision_is_ok():
    """Intended, not a bug (see the comment in strip_prose): a
    function whose only statement is a docstring and a function
    whose only statement is `pass` dump identically, so adding or
    deleting a sole docstring passes the guard either way.
    """
    before = _d('''
        def f():
            """doc"""
    ''')
    after = _d('''
        def f():
            pass
    ''')
    assert check(before, after)


# -- _verify_ref ---------------------------------------------------


def test_verify_ref_accepts_a_real_ref(repo):
    assert _verify_ref("HEAD") is None


def test_verify_ref_rejects_an_unresolvable_ref(repo):
    assert _verify_ref("NOSUCHREF") is not None


# -- _git_show -------------------------------------------------------


def test_git_show_returns_committed_content(repo):
    content, err = _git_show("HEAD", Path("a.py"))
    assert err is None
    assert content == "VALUE = 1\n"


def test_git_show_new_file_is_a_clean_skip(repo):
    (repo / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    content, err = _git_show("HEAD", Path("new.py"))
    assert content is None
    assert err is None


def test_git_show_other_failure_is_reported_not_skipped(repo):
    """A path absent both on disk and at the ref is not the "new
    file" case (git's message differs) and must come back as an
    error, never as a silent skip.
    """
    content, err = _git_show("HEAD", Path("nope.py"))
    assert content is None
    assert err is not None


# -- main(): the CLI, incl. the two reproduced Critical cases -------


def test_main_rejects_an_unresolvable_ref(repo, capsys):
    """Reproduces the Critical: a typo'd ref must never report OK."""
    rc = main(["NOSUCHREF", "a.py"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out


def test_main_rejects_a_path_matching_nothing_on_disk(repo, capsys):
    """Reproduces the Critical: a typo'd path must never report
    OK.
    """
    rc = main(["HEAD", "NOPE_TYPO.py"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out


def test_main_rejects_an_empty_directory(repo, capsys):
    (repo / "empty_dir").mkdir()
    rc = main(["HEAD", "empty_dir"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out


def test_main_rejects_an_empty_directory_mixed_with_a_comparison(
    repo, capsys
):
    """Discriminates the empty-directory guard from the unrelated
    "zero files compared" catch-all: a LONE empty-directory target
    is also caught by that catch-all (checked == 0 either way), so
    it doesn't prove the specific guard exists -- confirmed by
    mutation: deleting only the empty-dir check still left the
    lone-directory test above passing. Mixing in a genuine relative
    comparison removes that confound: without the guard, a.py alone
    would be compared, checked would be 1 (not 0), and the run would
    wrongly print "OK: 1 file(s)..." while silently never checking
    empty_dir.
    """
    (repo / "empty_dir").mkdir()
    rc = main(["HEAD", "a.py", "empty_dir"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out
    assert "empty_dir" in out


def test_main_all_new_files_is_an_error_not_ok(repo, capsys):
    """Every target being a brand-new file means zero files were
    actually compared; that must not print OK either.
    """
    (repo / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    rc = main(["HEAD", "new.py"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out


def test_main_happy_path_ok(repo, capsys):
    (repo / "a.py").write_text(
        "# reworded comment\nVALUE = 1\n", encoding="utf-8"
    )
    rc = main(["HEAD", "a.py"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: 1 file" in out


def test_main_detects_logic_change(repo, capsys):
    (repo / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    rc = main(["HEAD", "a.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "LOGIC CHANGED" in out


def test_main_new_file_alongside_a_real_comparison_is_ok(repo, capsys):
    """A new file mixed with at least one real comparison is fine;
    only an ALL-new run is an error.
    """
    (repo / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    rc = main(["HEAD", "a.py", "new.py"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: 1 file" in out
    assert "1 new" in out


def test_main_rejects_an_absolute_path(repo, capsys):
    """An absolute path is not supported input. git resolves the
    <ref>:<path> pathspec relative to the repo, so an absolute path
    to a real, unmodified, tracked file comes back from git as
    "exists on disk, but not in <ref>" -- indistinguishable from a
    genuinely new file, and would be silently skipped rather than
    compared (confirmed directly against git, not assumed). Reject
    it up front instead.
    """
    rc = main(["HEAD", str(repo / "a.py")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out


def test_main_rejects_an_absolute_path_mixed_with_a_comparison(
    repo, capsys
):
    """Discriminates the is_absolute() guard from the unrelated
    "zero files compared" catch-all: a LONE absolute-path target is
    also caught by that catch-all (checked == 0 either way, since
    git reports the absolute path as "exists on disk, but not in
    <ref>", the same skip-worthy message a genuine new file gets),
    so it doesn't prove the specific guard exists -- confirmed by
    mutation: deleting only the is_absolute() check still left the
    lone-absolute-path test above passing. Mixing in a genuine
    relative comparison removes that confound: without the guard,
    a.py alone would be compared, checked would be 1 (not 0), and
    the run would wrongly print "OK: 1 file(s)...(1 new)" while
    silently never comparing b.py.
    """
    _commit(repo, "b.py", "VALUE = 2\n", "add b")
    abs_b = str(repo / "b.py")
    rc = main(["HEAD", "a.py", abs_b])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK" not in out
    assert abs_b in out


def test_main_continues_after_a_git_show_hard_error(
    repo, capsys, monkeypatch
):
    """Any _git_show failure other than a genuine "new file" must be
    reported and must not silently skip -- and must not abort the
    rest of the run. The real git invocations that reach this branch
    are obscure (see test_git_show_other_failure_is_reported_not_
    skipped for one), so the branch is exercised directly here by
    faking one file's result, leaving the other file's real git call
    untouched.
    """
    _commit(repo, "b.py", "VALUE = 2\n", "add b")
    real_git_show = prose_guard._git_show

    def flaky(ref, path):
        if path.name == "a.py":
            return None, "simulated git failure"
        return real_git_show(ref, path)

    monkeypatch.setattr(prose_guard, "_git_show", flaky)
    rc = main(["HEAD", "a.py", "b.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "GIT ERROR" in out
    assert "simulated git failure" in out
    assert "LOGIC CHANGED" not in out


def test_main_continues_after_an_unreadable_file(repo, capsys):
    """A file that fails to decode as UTF-8 is reported and
    skipped, not a crash -- the rest of the files still get
    checked.
    """
    _commit(repo, "b.py", "VALUE = 2\n", "add b")
    (repo / "b.py").write_bytes(b"VALUE = \xff\xfe\n")
    rc = main(["HEAD", "a.py", "b.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "READ ERROR" in out
    assert "b.py" in out
    assert "LOGIC CHANGED" not in out


def test_main_continues_after_a_syntax_error(repo, capsys):
    _commit(repo, "b.py", "VALUE = 2\n", "add b")
    (repo / "b.py").write_text("def f(:\n", encoding="utf-8")
    rc = main(["HEAD", "a.py", "b.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "SYNTAX ERROR" in out
    assert "b.py" in out
    assert "LOGIC CHANGED" not in out
