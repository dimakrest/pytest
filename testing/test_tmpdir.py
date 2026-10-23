# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Callable
import dataclasses
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import cast
import warnings

from _pytest import pathlib
from _pytest.config import Config
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pathlib import _chmod_rwx
from _pytest.pathlib import cleanup_numbered_dir
from _pytest.pathlib import create_cleanup_lock
from _pytest.pathlib import make_numbered_dir
from _pytest.pathlib import maybe_delete_a_numbered_dir
from _pytest.pathlib import on_rm_rf_error
from _pytest.pathlib import register_cleanup_lock_removal
from _pytest.pathlib import rm_rf
from _pytest.pytester import Pytester
from _pytest.tmpdir import get_user
from _pytest.tmpdir import TempPathFactory
import pytest


def test_tmp_path_fixture(pytester: Pytester) -> None:
    p = pytester.copy_example("tmpdir/tmp_path_fixture.py")
    results = pytester.runpytest(p)
    results.stdout.fnmatch_lines(["*1 passed*"])


@dataclasses.dataclass
class FakeConfig:
    basetemp: str | Path

    @property
    def trace(self):
        return self

    def get(self, key):
        return lambda *k: None

    def getini(self, name):
        if name == "tmp_path_retention_count":
            return 3
        elif name == "tmp_path_retention_policy":
            return "all"
        else:
            assert False

    @property
    def option(self):
        return self


def patch_stat(monkeypatch: MonkeyPatch, path: Path, **change) -> None:
    """Make os.stat(path) report other values: each keyword is a stat field
    (st_mtime_ns, st_ctime_ns, st_nlink) and a function of its real value."""
    real_stat = os.stat

    def fake_stat(p, *args, **kwargs):
        st = real_stat(p, *args, **kwargs)
        if isinstance(p, int) or os.fspath(p) != os.fspath(path):
            return st
        values = list(st[:10])
        values[3] = change.get("st_nlink", lambda n: n)(st.st_nlink)
        times = {
            f: change.get(f, lambda t: t)(getattr(st, f))
            for f in ("st_atime_ns", "st_mtime_ns", "st_ctime_ns")
        }
        return os.stat_result(values, times)

    monkeypatch.setattr(os, "stat", fake_stat)


def count_listings(monkeypatch: MonkeyPatch, path: Path) -> list[int]:
    """Count the os.scandir(path) calls from now on."""
    calls = [0]
    real_scandir = os.scandir

    def scandir(p="."):
        if not isinstance(p, int) and os.fspath(p) == os.fspath(path):
            calls[0] += 1
        return real_scandir(p)

    monkeypatch.setattr(os, "scandir", scandir)
    return calls


def wait_for_clock_tick(path: Path, probe: Path) -> None:
    """Wait until a change to the directory path gets a later timestamp than
    its last one, by creating the directory probe elsewhere.

    With a coarse filesystem clock, a change in the same tick as mktemp's last
    look at the directory may not show in its stat. That limit is documented
    and not what the tests using this are about.
    """
    mtime = os.stat(path).st_mtime_ns
    for _ in range(1000):
        probe.mkdir()
        later = os.stat(probe).st_mtime_ns > mtime
        probe.rmdir()
        if later:
            return
        time.sleep(0.01)
    pytest.fail("the filesystem clock did not advance in 10 seconds")


class TestTmpPathHandler:
    def test_mktemp(self, tmp_path: Path) -> None:
        config = cast(Config, FakeConfig(tmp_path))
        t = TempPathFactory.from_config(config, _ispytest=True)
        tmp = t.mktemp("world")
        assert str(tmp.relative_to(t.getbasetemp())) == "world0"
        tmp = t.mktemp("this")
        assert str(tmp.relative_to(t.getbasetemp())).startswith("this")
        tmp2 = t.mktemp("this")
        assert str(tmp2.relative_to(t.getbasetemp())).startswith("this")
        assert tmp2 != tmp

    def test_tmppath_relative_basetemp_absolute(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """#4425"""
        monkeypatch.chdir(tmp_path)
        config = cast(Config, FakeConfig("hello"))
        t = TempPathFactory.from_config(config, _ispytest=True)
        assert t.getbasetemp().resolve() == (tmp_path / "hello").resolve()


class TestMktempNumbering:
    """mktemp keeps the listing of the base temp directory between calls. The
    names must still be those that a fresh listing on every call gives."""

    @pytest.fixture
    def factory(self, tmp_path: Path) -> TempPathFactory:
        config = cast(Config, FakeConfig(tmp_path / "basetemp"))
        return TempPathFactory.from_config(config, _ispytest=True)

    def test_lists_basetemp_once(self, factory, monkeypatch):
        listings = count_listings(monkeypatch, factory.getbasetemp())
        names = [factory.mktemp("world").name for _ in range(20)]
        assert names == [f"world{i}" for i in range(20)]
        # Not always 1: where the timestamps or st_nlink cannot show our own
        # change reliably, the next call lists again.
        assert listings[0] < 10

    def test_numbering_restarts_when_basetemp_is_reused(self, tmp_path, monkeypatch):
        config = cast(Config, FakeConfig(tmp_path / "basetemp"))
        factory = TempPathFactory.from_config(config, _ispytest=True)
        for _ in range(2):
            assert [factory.mktemp("world").name for _ in range(3)] == [
                "world0",
                "world1",
                "world2",
            ]
            # The same --basetemp again, as pytester.inline_run uses it.
            factory = TempPathFactory.from_config(config, _ispytest=True)
        factory.mktemp("world")
        # The same factory setting up its basetemp again.
        monkeypatch.setattr(factory, "_basetemp", None)
        assert factory.mktemp("world").name == "world0"

    def test_sees_changes_by_others(self, factory, tmp_path):
        basetemp = factory.getbasetemp()
        factory.mktemp("fun-")
        wait_for_clock_tick(basetemp, tmp_path / "tick")
        basetemp.joinpath("fun-7").mkdir()
        assert factory.mktemp("fun-").name == "fun-8"
        # A file does not change st_nlink, only the timestamps.
        wait_for_clock_tick(basetemp, tmp_path / "tick")
        basetemp.joinpath("fun-11").touch()
        assert factory.mktemp("fun-").name == "fun-12"
        # A removed number is used again, as with a fresh listing.
        wait_for_clock_tick(basetemp, tmp_path / "tick")
        basetemp.joinpath("fun-12").rmdir()
        basetemp.joinpath("fun-11").unlink()
        assert factory.mktemp("fun-").name == "fun-9"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="st_ctime is the creation time on Windows"
    )
    def test_sees_change_with_mtime_set_back(self, factory, tmp_path):
        basetemp = factory.getbasetemp()
        factory.mktemp("fun-")
        st = os.stat(basetemp)
        wait_for_clock_tick(basetemp, tmp_path / "tick")
        basetemp.joinpath("fun-5").touch()
        # Restore the old mtime, as touch -r or rsync -t do; st_ctime changes.
        os.utime(basetemp, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert factory.mktemp("fun-").name == "fun-6"

    def test_sees_change_with_frozen_timestamps(self, factory, monkeypatch):
        # The timestamps can miss a change (coarse clock); a new subdirectory
        # still shows in st_nlink where it counts them.
        basetemp = factory.getbasetemp()
        if os.stat(basetemp).st_nlink < 2:
            pytest.skip("st_nlink does not count subdirectories here")
        frozen = lambda t: 1_234_567_891  # noqa: E731
        patch_stat(monkeypatch, basetemp, st_mtime_ns=frozen, st_ctime_ns=frozen)
        factory.mktemp("fun-")
        basetemp.joinpath("fun-7").mkdir()
        assert factory.mktemp("fun-").name == "fun-8"
        # Also when it happens while mktemp creates its own directory.
        real_mkdir = Path.mkdir
        foreign = [basetemp / "fun-20"]

        def mkdir(self, *args, **kwargs):
            while foreign:
                real_mkdir(foreign.pop())
            real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir)
        assert factory.mktemp("fun-").name == "fun-9"
        assert factory.mktemp("fun-").name == "fun-21"

    def test_lists_again_when_number_is_taken(self, factory, monkeypatch):
        # A change that the stat does not show: the next number is taken.
        basetemp = factory.getbasetemp()
        frozen = lambda t: 1_234_567_891  # noqa: E731
        patch_stat(
            monkeypatch,
            basetemp,
            st_mtime_ns=frozen,
            st_ctime_ns=frozen,
            st_nlink=lambda n: 1,
        )
        factory.mktemp("fun-")
        basetemp.joinpath("fun-1").mkdir()
        assert factory.mktemp("fun-").name == "fun-2"

    def test_sees_change_while_listing(self, factory, tmp_path, monkeypatch):
        basetemp = factory.getbasetemp()
        real_scandir = os.scandir

        def scandir(p="."):
            entries = list(real_scandir(p))
            wait_for_clock_tick(basetemp, tmp_path / "tick")
            basetemp.joinpath("fun-30").touch()
            monkeypatch.setattr(os, "scandir", real_scandir)
            return iter(entries)

        monkeypatch.setattr(os, "scandir", scandir)
        assert factory.mktemp("fun-").name == "fun-0"
        assert factory.mktemp("fun-").name == "fun-31"

    @pytest.mark.parametrize("seconds", [1, 2])
    def test_lists_every_time_with_coarse_timestamps(
        self, factory, monkeypatch, seconds
    ):
        # Filesystems with 1 or 2 second timestamps (FAT, HFS+, ext3).
        basetemp = factory.getbasetemp()
        step = seconds * 1_000_000_000
        coarse = lambda t: t // step * step  # noqa: E731
        patch_stat(monkeypatch, basetemp, st_mtime_ns=coarse, st_ctime_ns=coarse)
        listings = count_listings(monkeypatch, basetemp)
        for i in range(3):
            assert factory.mktemp("fun-").name == f"fun-{i}"
            basetemp.joinpath(f"other{i}").touch()
        assert listings[0] == 3

    def test_case_insensitive(self, factory):
        assert factory.mktemp("test_a").name == "test_a0"
        assert factory.mktemp("test_A").name == "test_A1"
        assert factory.mktemp("test_a").name == "test_a2"

    def test_overlapping_names(self, factory):
        # The number is everything after the name, so test_a10 is number 10
        # of test_a, and test_a1 is number 0 of test_a1.
        for _ in range(3):
            factory.mktemp("test_a")
        for _ in range(2):
            factory.mktemp("test_a1")
        assert factory.mktemp("test_a").name == "test_a12"
        assert factory.mktemp("test_a1").name == "test_a13"

    def test_names_ending_in_digits(self, tmp_path):
        for name in ["test_v0", "test_v1", "test_v2", "test_v20", "test_v21"]:
            tmp_path.joinpath(name).mkdir()
        factory = TempPathFactory(
            None, 3, "all", lambda *args: None, basetemp=tmp_path, _ispytest=True
        )
        assert factory.mktemp("test_v2").name == "test_v22"
        assert factory.mktemp("test_v").name == "test_v23"


class TestConfigTmpPath:
    def test_getbasetemp_custom_removes_old(self, pytester: Pytester) -> None:
        mytemp = pytester.path.joinpath("xyz")
        p = pytester.makepyfile(
            """
            def test_1(tmp_path):
                pass
        """
        )
        pytester.runpytest(p, f"--basetemp={mytemp}")
        assert mytemp.exists()
        mytemp.joinpath("hello").touch()

        pytester.runpytest(p, f"--basetemp={mytemp}")
        assert mytemp.exists()
        assert not mytemp.joinpath("hello").exists()

    def test_policy_none_delete_all(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            """
            def test_1(tmp_path):
                assert 0 == 0
        """
        )
        p_failed = pytester.makepyfile(
            another_file_name="""
            def test_1(tmp_path):
                assert 0 == 1
        """
        )
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "none"
        """
        )

        pytester.inline_run(p)
        pytester.inline_run(p_failed)

        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(child.iterdir())
            # Check the base dir itself is gone without depending on test results
            assert base_dir == []

    @pytest.mark.parametrize("policy", ['"failed"', '"all"'])
    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_retention_count(self, pytester: Pytester, policy, count) -> None:
        p = pytester.makepyfile(
            """
            def test_1(tmp_path):
                assert 0 == 0
        """
        )
        p_failed = pytester.makepyfile(
            another_file_name="""
            def test_1(tmp_path):
                assert 0 == 1
        """
        )

        pytester.makepyprojecttoml(
            f"""
            [tool.pytest.ini_options]
            tmp_path_retention_policy = {policy}
            tmp_path_retention_count = {count}
        """
        )

        pytester.inline_run(p)
        pytester.inline_run(p_failed)
        pytester.inline_run(p)
        pytester.inline_run(p_failed)
        pytester.inline_run(p)
        pytester.inline_run(p_failed)
        pytester.inline_run(p)
        pytester.inline_run(p_failed)

        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = filter(lambda x: not x.is_symlink(), child.iterdir())
            assert len(list(base_dir)) == count

    def test_policy_failed_removes_only_passed_dir(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            """
            def test_1(tmp_path):
                assert 0 == 0
            def test_2(tmp_path):
                assert 0 == 1
        """
        )
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "failed"
        """
        )

        pytester.inline_run(p)
        root = pytester._test_tmproot

        for child in root.iterdir():
            base_dir = list(
                filter(lambda x: x.is_dir() and not x.is_symlink(), child.iterdir())
            )
            assert len(base_dir) == 1
            test_dir = list(
                filter(
                    lambda x: x.is_dir() and not x.is_symlink(), base_dir[0].iterdir()
                )
            )
            # Check only the failed one remains
            assert len(test_dir) == 1
            assert test_dir[0].name == "test_20"

    def test_policy_failed_removes_basedir_when_all_passed(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
            """
            def test_1(tmp_path):
                assert 0 == 0
        """
        )
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "failed"
        """
        )

        pytester.inline_run(p)
        root = pytester._test_tmproot
        for child in root.iterdir():
            # This symlink will be deleted by cleanup_numbered_dir **after**
            # the test finishes because it's triggered by atexit.
            # So it has to be ignored here.
            base_dir = filter(lambda x: not x.is_symlink(), child.iterdir())
            # Check the base dir itself is gone
            assert len(list(base_dir)) == 0

    def test_policy_failed_numbering_unseen_by_stat(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        # Where the base temp directory's stat does not show pytest's own
        # changes to it (Windows: a coarse clock and st_nlink 1), the names must
        # still be those of a fresh listing.
        basetemp = pytester.path / "basetemp"
        frozen = lambda t: 1_234_567_891  # noqa: E731
        patch_stat(
            monkeypatch,
            basetemp,
            st_mtime_ns=frozen,
            st_ctime_ns=frozen,
            st_nlink=lambda n: 1,
        )
        listings = count_listings(monkeypatch, basetemp)
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "failed"
        """
        )
        same_name = "def test_x(tmp_path):\n    assert tmp_path.name == {!r}\n"
        pytester.makepyfile(
            test_a=same_name.format("test_x0"),
            test_b=same_name.format("test_x0"),
            test_c="""
            def test_y(tmp_path_factory):
                tmp_path_factory.mktemp("test_x5", numbered=False)
            """,
            test_d=same_name.format("test_x6"),
        )
        pytester.inline_run(f"--basetemp={basetemp}").assertoutcome(passed=4)
        # Once by mktemp; possibly once more at the end of the session.
        assert listings[0] <= 2

    # issue #10502
    def test_policy_failed_removes_dir_when_skipped_from_fixture(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
            """
            import pytest

            @pytest.fixture
            def fixt(tmp_path):
                pytest.skip()

            def test_fixt(fixt):
                pass
        """
        )
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "failed"
        """
        )

        pytester.inline_run(p)

        # Check if the whole directory is removed
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(child.iterdir())
            assert base_dir == []

    # issue #10502
    def test_policy_all_keeps_dir_when_skipped_from_fixture(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
            """
            import pytest

            @pytest.fixture
            def fixt(tmp_path):
                pytest.skip()

            def test_fixt(fixt):
                pass
        """
        )
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            tmp_path_retention_policy = "all"
        """
        )
        pytester.inline_run(p)

        # Check if the whole directory is kept
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(
                filter(lambda x: x.is_dir() and not x.is_symlink(), child.iterdir())
            )
            assert len(base_dir) == 1
            test_dir = list(
                filter(
                    lambda x: x.is_dir() and not x.is_symlink(), base_dir[0].iterdir()
                )
            )
            assert len(test_dir) == 1


testdata = [
    ("mypath", True),
    ("/mypath1", False),
    ("./mypath1", True),
    ("../mypath3", False),
    ("../../mypath4", False),
    ("mypath5/..", False),
    ("mypath6/../mypath6", True),
    ("mypath7/../mypath7/..", False),
]


@pytest.mark.parametrize("basename, is_ok", testdata)
def test_mktemp(pytester: Pytester, basename: str, is_ok: bool) -> None:
    mytemp = pytester.mkdir("mytemp")
    p = pytester.makepyfile(
        f"""
        def test_abs_path(tmp_path_factory):
            tmp_path_factory.mktemp('{basename}', numbered=False)
        """
    )

    result = pytester.runpytest(p, f"--basetemp={mytemp}")
    if is_ok:
        assert result.ret == 0
        assert mytemp.joinpath(basename).exists()
    else:
        assert result.ret == 1
        result.stdout.fnmatch_lines("*ValueError*")


def test_tmp_path_always_is_realpath(pytester: Pytester, monkeypatch) -> None:
    # the reason why tmp_path should be a realpath is that
    # when you cd to it and do "os.getcwd()" you will anyway
    # get the realpath.  Using the symlinked path can thus
    # easily result in path-inequality
    # XXX if that proves to be a problem, consider using
    # os.environ["PWD"]
    realtemp = pytester.mkdir("myrealtemp")
    linktemp = pytester.path.joinpath("symlinktemp")
    attempt_symlink_to(linktemp, str(realtemp))
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(linktemp))
    pytester.makepyfile(
        """
        def test_1(tmp_path):
            assert tmp_path.resolve() == tmp_path
    """
    )
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)


def test_tmp_path_too_long_on_parametrization(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.parametrize("arg", ["1"*1000])
        def test_some(arg, tmp_path):
            tmp_path.joinpath("hello").touch()
    """
    )
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)


def test_tmp_path_factory(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.fixture(scope='session')
        def session_dir(tmp_path_factory):
            return tmp_path_factory.mktemp('data', numbered=False)
        def test_some(session_dir):
            assert session_dir.is_dir()
    """
    )
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)


def test_tmp_path_fallback_tox_env(pytester: Pytester, monkeypatch) -> None:
    """Test that tmp_path works even if environment variables required by getpass
    module are missing (#1010).
    """
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    pytester.makepyfile(
        """
        def test_some(tmp_path):
            assert tmp_path.is_dir()
    """
    )
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)


@pytest.fixture
def break_getuser(monkeypatch):
    monkeypatch.setattr("os.getuid", lambda: -1)
    # taken from python 2.7/3.4
    for envvar in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(envvar, raising=False)


@pytest.mark.usefixtures("break_getuser")
@pytest.mark.skipif(sys.platform.startswith("win"), reason="no os.getuid on windows")
def test_tmp_path_fallback_uid_not_found(pytester: Pytester) -> None:
    """Test that tmp_path works even if the current process's user id does not
    correspond to a valid user.
    """
    pytester.makepyfile(
        """
        def test_some(tmp_path):
            assert tmp_path.is_dir()
    """
    )
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)


@pytest.mark.usefixtures("break_getuser")
@pytest.mark.skipif(sys.platform.startswith("win"), reason="no os.getuid on windows")
def test_get_user_uid_not_found():
    """Test that get_user() function works even if the current process's
    user id does not correspond to a valid user (e.g. running pytest in a
    Docker container with 'docker run -u'.
    """
    assert get_user() is None


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="win only")
def test_get_user(monkeypatch):
    """Test that get_user() function works even if environment variables
    required by getpass module are missing from the environment on Windows
    (#1010).
    """
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    assert get_user() is None


class TestNumberedDir:
    PREFIX = "fun-"

    def test_make(self, tmp_path):
        for i in range(10):
            d = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)
            assert d.name.startswith(self.PREFIX)
            assert d.name.endswith(str(i))

        symlink = tmp_path.joinpath(self.PREFIX + "current")
        if symlink.exists():
            # unix
            assert symlink.is_symlink()
            assert symlink.resolve() == d.resolve()

    def test_cleanup_lock_create(self, tmp_path):
        d = tmp_path.joinpath("test")
        d.mkdir()
        lockfile = create_cleanup_lock(d)
        with pytest.raises(OSError, match=r"cannot create lockfile in .*"):
            create_cleanup_lock(d)

        lockfile.unlink()

    def test_lock_register_cleanup_removal(self, tmp_path: Path) -> None:
        lock = create_cleanup_lock(tmp_path)

        registry: list[Callable[..., None]] = []
        register_cleanup_lock_removal(lock, register=registry.append)

        (cleanup_func,) = registry

        assert lock.is_file()

        cleanup_func(original_pid="intentionally_different")

        assert lock.is_file()

        cleanup_func()

        assert not lock.exists()

        cleanup_func()

        assert not lock.exists()

    def _do_cleanup(self, tmp_path: Path, keep: int = 2) -> None:
        self.test_make(tmp_path)
        cleanup_numbered_dir(
            root=tmp_path,
            prefix=self.PREFIX,
            keep=keep,
            consider_lock_dead_if_created_before=0,
        )

    def test_cleanup_keep(self, tmp_path):
        self._do_cleanup(tmp_path)
        a, b = (x for x in tmp_path.iterdir() if not x.is_symlink())
        print(a, b)

    def test_cleanup_keep_0(self, tmp_path: Path):
        self._do_cleanup(tmp_path, 0)
        dir_num = len(list(tmp_path.iterdir()))
        assert dir_num == 0

    def test_cleanup_locked(self, tmp_path):
        p = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)

        create_cleanup_lock(p)

        assert not pathlib.ensure_deletable(
            p, consider_lock_dead_if_created_before=p.stat().st_mtime - 1
        )
        assert pathlib.ensure_deletable(
            p, consider_lock_dead_if_created_before=p.stat().st_mtime + 1
        )

    def test_cleanup_ignores_symlink(self, tmp_path):
        the_symlink = tmp_path / (self.PREFIX + "current")
        attempt_symlink_to(the_symlink, tmp_path / (self.PREFIX + "5"))
        self._do_cleanup(tmp_path)

    def test_removal_accepts_lock(self, tmp_path):
        dir = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)
        create_cleanup_lock(dir)
        maybe_delete_a_numbered_dir(dir)
        assert dir.is_dir()


def _raise_oserror(*args: object, **kwargs: object) -> None:
    raise OSError("simulated failure")


class TestRmRf:
    def test_rm_rf(self, tmp_path):
        adir = tmp_path / "adir"
        adir.mkdir()
        rm_rf(adir)

        assert not adir.exists()

        adir.mkdir()
        afile = adir / "afile"
        afile.write_bytes(b"aa")

        rm_rf(adir)
        assert not adir.exists()

    def test_rm_rf_with_read_only_file(self, tmp_path):
        """Ensure rm_rf can remove directories with read-only files in them (#5524)"""
        fn = tmp_path / "dir/foo.txt"
        fn.parent.mkdir()

        fn.touch()

        self.chmod_r(fn)

        rm_rf(fn.parent)

        assert not fn.parent.is_dir()

    def chmod_r(self, path):
        mode = os.stat(str(path)).st_mode
        os.chmod(str(path), mode & ~stat.S_IWRITE)

    def test_rm_rf_with_read_only_directory(self, tmp_path):
        """Ensure rm_rf can remove read-only directories (#5524)"""
        adir = tmp_path / "dir"
        adir.mkdir()

        (adir / "foo.txt").touch()
        self.chmod_r(adir)

        rm_rf(adir)

        assert not adir.is_dir()

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="unix permissions")
    def test_rm_rf_with_no_exec_permission_directories(self, tmp_path):
        """Ensure rm_rf can remove directories without S_IXUSR (#7940).

        This is the exact scenario from the original issue: nested directories
        and files with all permissions stripped.
        """
        p = tmp_path / "foo" / "bar" / "baz"
        p.parent.mkdir(parents=True)
        p.touch(mode=0)
        for parent in p.parents:  # pragma: no branch
            if parent == tmp_path:
                break
            parent.chmod(mode=0)

        rm_rf(tmp_path / "foo")

        assert not (tmp_path / "foo").exists()

    def test_on_rm_rf_error(self, tmp_path: Path) -> None:
        adir = tmp_path / "dir"
        adir.mkdir()

        fn = adir / "foo.txt"
        fn.touch()
        self.chmod_r(fn)

        # unknown exception
        with pytest.warns(pytest.PytestWarning):
            exc_info1 = (RuntimeError, RuntimeError(), None)
            on_rm_rf_error(os.unlink, str(fn), exc_info1, start_path=tmp_path)
            assert fn.is_file()

        # we ignore FileNotFoundError
        exc_info2 = (FileNotFoundError, FileNotFoundError(), None)
        assert not on_rm_rf_error(None, str(fn), exc_info2, start_path=tmp_path)

        # unknown function
        with pytest.warns(
            pytest.PytestWarning,
            match=r"^\(rm_rf\) unknown function None when removing .*foo.txt:\n<class 'PermissionError'>: ",
        ):
            exc_info3 = (PermissionError, PermissionError(), None)
            on_rm_rf_error(None, str(fn), exc_info3, start_path=tmp_path)
            assert fn.is_file()

        # os.unlink PermissionError is handled (chmod + retry)
        exc_info5 = PermissionError()
        on_rm_rf_error(os.unlink, str(fn), exc_info5, start_path=tmp_path)
        assert not fn.is_file()

    def test_on_rm_rf_error_os_open_handles_file(self, tmp_path: Path) -> None:
        """os.open PermissionError on a file is handled by fixing
        permissions and removing it (#7940)."""
        adir = tmp_path / "dir"
        adir.mkdir()
        fn = adir / "foo.txt"
        fn.touch()
        self.chmod_r(fn)

        with warnings.catch_warnings(record=True) as w:
            exc_info = PermissionError()
            on_rm_rf_error(os.open, str(fn), exc_info, start_path=tmp_path)
            assert not fn.exists()
            assert not [x.message for x in w]

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="unix permissions")
    def test_on_rm_rf_error_os_open_handles_directory(self, tmp_path: Path) -> None:
        """os.open PermissionError on a directory is handled by fixing
        permissions and recursively removing it (#7940)."""
        adir = tmp_path / "dir"
        adir.mkdir()
        (adir / "child").mkdir()
        (adir / "child" / "file.txt").touch()
        os.chmod(str(adir), stat.S_IRUSR | stat.S_IWUSR)

        with warnings.catch_warnings(record=True) as w:
            exc_info = PermissionError()
            on_rm_rf_error(os.open, str(adir), exc_info, start_path=tmp_path)
            assert not adir.exists()
            assert not [x.message for x in w]

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="unix permissions")
    def test_on_rm_rf_error_os_open_parent_perms(self, tmp_path: Path) -> None:
        """When the PermissionError is caused by the *parent* directory lacking
        S_IXUSR, fixing the parent is sufficient even if the child already has
        correct permissions."""
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        (child / "file.txt").touch()
        # Child has full perms, but parent lacks execute -> os.open(child) fails.
        os.chmod(str(parent), stat.S_IRUSR | stat.S_IWUSR)

        with warnings.catch_warnings(record=True) as w:
            exc_info = PermissionError()
            result = on_rm_rf_error(os.open, str(child), exc_info, start_path=tmp_path)
            assert result is True
            assert not child.exists()
            assert not [x.message for x in w]

    def test_chmod_rwx_returns_false_on_nonexistent_path(self, tmp_path: Path) -> None:
        """_chmod_rwx returns False when the path doesn't exist (OSError)."""
        nonexistent = tmp_path / "does_not_exist"
        assert _chmod_rwx(str(nonexistent)) is False

    def test_chmod_rwx_returns_false_when_chmod_fails(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """_chmod_rwx returns False when os.chmod raises OSError.

        A real FS layout that makes chmod fail while the path remains
        (immutable bit, RO mount, foreign ownership) is not portable in CI,
        so pin this branch with a monkeypatch.
        """
        fn = tmp_path / "file.txt"
        fn.touch(mode=0)  # needs bits so we reach os.chmod
        monkeypatch.setattr(os, "chmod", _raise_oserror)
        assert _chmod_rwx(str(fn)) is False

    def test_chmod_rwx_returns_false_when_already_sufficient(
        self, tmp_path: Path
    ) -> None:
        """_chmod_rwx returns False when permissions are already sufficient."""
        d = tmp_path / "dir"
        d.mkdir(mode=stat.S_IRWXU)
        assert _chmod_rwx(str(d)) is False

        f = tmp_path / "file"
        f.touch(mode=stat.S_IRUSR | stat.S_IWUSR)
        assert _chmod_rwx(str(f)) is False

    def test_on_rm_rf_error_os_open_returns_false_when_chmod_ineffective(
        self, tmp_path: Path
    ) -> None:
        """os.open handler returns False when neither parent nor path chmod
        changes anything (recursion guard)."""
        adir = tmp_path / "dir"
        adir.mkdir(mode=stat.S_IRWXU)
        exc_info = PermissionError()
        result = on_rm_rf_error(os.open, str(adir), exc_info, start_path=tmp_path)
        assert result is False
        assert adir.exists()

    def test_on_rm_rf_error_os_open_unlink_fails(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """os.open handler returns False when chmod succeeds but os.unlink
        still raises OSError (e.g. sandbox or other mechanism)."""
        fn = tmp_path / "stubborn.txt"
        fn.touch(mode=0)

        monkeypatch.setattr(os, "unlink", _raise_oserror)

        exc_info = PermissionError()
        result = on_rm_rf_error(os.open, str(fn), exc_info, start_path=tmp_path)
        assert result is False
        assert fn.exists()

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="unix permissions")
    def test_on_rm_rf_error_chmod_retry_walks_parents(self, tmp_path: Path) -> None:
        """The os.rmdir/os.unlink handler walks up through multiple parent
        directories to fix permissions before retrying."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        fn = deep / "file.txt"
        fn.touch()
        # Remove write from intermediate dirs (keep exec so traversal works,
        # but os.unlink needs write on the parent).
        for parent in fn.parents:  # pragma: no branch
            if parent == tmp_path:
                break
            parent.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )

        exc_info = PermissionError()
        on_rm_rf_error(os.unlink, str(fn), exc_info, start_path=tmp_path)
        assert not fn.exists()


def attempt_symlink_to(path, to_path):
    """Try to make a symlink from "path" to "to_path", skipping in case this platform
    does not support it or we don't have sufficient privileges (common on Windows)."""
    try:
        Path(path).symlink_to(Path(to_path))
    except OSError:
        pytest.skip("could not create symbolic link")


def test_basetemp_with_read_only_files(pytester: Pytester) -> None:
    """Integration test for #5524"""
    pytester.makepyfile(
        """
        import os
        import stat

        def test(tmp_path):
            fn = tmp_path / 'foo.txt'
            fn.write_text('hello', encoding='utf-8')
            mode = os.stat(str(fn)).st_mode
            os.chmod(str(fn), mode & ~stat.S_IREAD)
    """
    )
    result = pytester.runpytest("--basetemp=tmp")
    assert result.ret == 0
    # running a second time and ensure we don't crash
    result = pytester.runpytest("--basetemp=tmp")
    assert result.ret == 0


def test_tmp_path_factory_handles_invalid_dir_characters(
    tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "os/<:*?;>agnostic")
    # _basetemp / _given_basetemp are cached / set in parallel runs, patch them
    monkeypatch.setattr(tmp_path_factory, "_basetemp", None)
    monkeypatch.setattr(tmp_path_factory, "_given_basetemp", None)
    p = tmp_path_factory.getbasetemp()
    assert "pytest-of-unknown" in str(p)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="checks unix permissions")
def test_tmp_path_factory_create_directory_with_safe_permissions(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify that pytest creates directories under /tmp with private permissions."""
    # Use the test's tmp_path as the system temproot (/tmp).
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path))
    tmp_factory = TempPathFactory(None, 3, "all", lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()

    # No world-readable permissions.
    assert (basetemp.stat().st_mode & 0o077) == 0
    # Parent too (pytest-of-foo).
    assert (basetemp.parent.stat().st_mode & 0o077) == 0


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="checks unix permissions")
def test_tmp_path_factory_fixes_up_world_readable_permissions(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify that if a /tmp/pytest-of-foo directory already exists with
    world-readable permissions, it is fixed.

    pytest used to mkdir with such permissions, that's why we fix it up.
    """
    # Use the test's tmp_path as the system temproot (/tmp).
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path))
    tmp_factory = TempPathFactory(None, 3, "all", lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()

    # Before - simulate bad perms.
    os.chmod(basetemp.parent, 0o777)
    assert (basetemp.parent.stat().st_mode & 0o077) != 0

    tmp_factory = TempPathFactory(None, 3, "all", lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()

    # After - fixed.
    assert (basetemp.parent.stat().st_mode & 0o077) == 0


@pytest.mark.skipif(
    not hasattr(os, "getuid") or os.stat not in os.supports_follow_symlinks,
    reason="checks unix permissions and symlinks",
)
def test_tmp_path_factory_doesnt_follow_symlinks(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify that if a /tmp/pytest-of-foo directory is a symbolic link,
    it is rejected (#13669, CVE-2025-71176)."""
    attacker_controlled = tmp_path / "attacker_controlled"
    attacker_controlled.mkdir()

    # Use the test's tmp_path as the system temproot (/tmp).
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path))

    # First just get the pytest-of-user path.
    tmp_factory = TempPathFactory(None, 3, "all", lambda *args: None, _ispytest=True)
    pytest_of_user = tmp_factory.getbasetemp().parent
    # Just for safety in the test, before we nuke it.
    assert "pytest-of-" in str(pytest_of_user)
    shutil.rmtree(pytest_of_user)

    pytest_of_user.symlink_to(attacker_controlled)

    # This now tries to use the directory when it's a symlink.
    tmp_factory = TempPathFactory(None, 3, "all", lambda *args: None, _ispytest=True)
    with pytest.raises(OSError, match=r"temporary directory .* is a symbolic link"):
        tmp_factory.getbasetemp()


def test_get_user_handles_getpass_oserror(monkeypatch: MonkeyPatch) -> None:
    """Regression test: get_user() should return None when getpass.getuser()
    raises OSError (Python 3.13+ behavior, #13835)."""
    import getpass

    def _raise_oserror():
        raise OSError("No username set in the environment")

    monkeypatch.setattr(getpass, "getuser", _raise_oserror)
    assert get_user() is None


def test_tmp_path_retention_policy_invalid(pytester: Pytester) -> None:
    """An invalid tmp_path_retention_policy fails with a clean usage error."""
    pytester.makepyprojecttoml(
        """
        [tool.pytest]
        tmp_path_retention_policy = "compress"
        """
    )
    pytester.makepyfile("def test(): pass")
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        [
            "*ERROR: *config option 'tmp_path_retention_policy' expects one of "
            "'all' | 'failed' | 'none', got 'compress'"
        ]
    )
