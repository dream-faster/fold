import pytest
from mktestdocs import check_docstring, get_codeblock_members

from fold import transformations

# This retrieves all methods/properties that have a docstring.
members = get_codeblock_members(transformations)


@pytest.mark.parametrize("obj", members, ids=lambda d: d.__name__)
def test_member(obj):
    check_docstring(obj, lang="bash")
