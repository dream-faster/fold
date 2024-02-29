# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from ...base.classes import Backend
from ..types import BackendType


def get_backend(backend_type: str | BackendType | Backend) -> Backend:
    if isinstance(backend_type, str):
        backend_type = BackendType.from_str(backend_type)

    if isinstance(backend_type, Backend):
        return backend_type

    if backend_type == BackendType.ray:
        from .ray import RayBackend

        return RayBackend()
    if backend_type == BackendType.no:
        from .sequential import NoBackend

        return NoBackend()
    if backend_type == BackendType.pathos:
        from .pathos import PathosBackend

        return PathosBackend()
    if backend_type == BackendType.thread:
        from .thread import ThreadBackend

        return ThreadBackend()
    if backend_type == BackendType.joblib:
        from .joblib import JoblibBackend

        return JoblibBackend()

    raise ValueError(f"Backend type {backend_type} not supported.")
