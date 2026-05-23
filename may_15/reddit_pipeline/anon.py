"""Stable, salted user-ID anonymization.

We never store raw Reddit usernames in extracted output. A user's pseudonymous
ID is `anon_<sha256(salt || username)[:16]>`. The salt is read from
ANON_SALT env var; without it the hash is still one-way but predictable across
researchers running on the same data, which is what we want for joinability.
"""

from __future__ import annotations

import hashlib
import os

_SALT = os.environ.get("ANON_SALT", "").encode()


def anon_user_id(username: str) -> str:
    h = hashlib.sha256(_SALT + username.encode("utf-8")).hexdigest()[:16]
    return f"anon_{h}"
