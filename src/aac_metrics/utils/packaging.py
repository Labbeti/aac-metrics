#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from dataclasses import dataclass, asdict, astuple
from functools import cache
from importlib.util import find_spec
from typing import Any, ClassVar, Mapping


@cache
def _package_is_available(package_name: str) -> bool:
    """Returns True if package is installed in the current python environment."""
    try:
        return find_spec(package_name) is not None
    except AttributeError:
        # Python 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python 3.7+
        return False


@dataclass(init=True, repr=True, eq=True, slots=True)
class Version:
    _VERSION_FORMAT: ClassVar[str] = "{major}.{minor}.{patch}"
    _VERSION_PATTERN: ClassVar[
        str
    ] = r"(?P<major>[^\.]+)\.(?P<minor>[^\.]+)\.(?P<patch>[^\.]+).*"

    major: str
    minor: str
    patch: str

    def __init__(self, major: Any, minor: Any, patch: Any) -> None:
        major = str(major)
        minor = str(minor)
        patch = str(patch)

        self.major = major
        self.minor = minor
        self.patch = patch

    @classmethod
    def from_dict(cls, version: Mapping[str, Any]) -> "Version":
        major = version["major"]
        minor = version["minor"]
        patch = version["patch"]
        return Version(major, minor, patch)

    @classmethod
    def from_str(cls, version: str) -> "Version":
        matched = re.match(Version._VERSION_PATTERN, version)
        if matched is None:
            raise ValueError(
                f"Invalid argument {version=}. (does not match pattern {Version._VERSION_PATTERN})"
            )
        matched_dict = matched.groupdict()
        return cls.from_dict(matched_dict)

    @classmethod
    def from_tuple(cls, version: tuple[Any, Any, Any]) -> "Version":
        major = version[0]
        minor = version[1]
        patch = version[2]
        return Version(major, minor, patch)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_str(self) -> str:
        return Version._VERSION_FORMAT.format(**self.to_dict())

    def to_tuple(self) -> tuple[str, str, str]:
        return astuple(self)  # type: ignore

    def __lt__(self, other: "Version") -> bool:
        if self.major < other.major:
            return True
        if self.major == other.major and self.minor < other.minor:
            return True
        if (
            self.major == other.major
            and self.minor == other.minor
            and self.patch < other.patch
        ):
            return True
        return False

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __ge__(self, other: "Version") -> bool:
        return not (self < other)

    def __gt__(self, other: "Version") -> bool:
        return not (self <= other)
