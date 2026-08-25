import pytensor_ml

from tests.public_api import all_docstrings, attribute_docstrings, public_objects

# Both docstring sweeps parametrize over these collectors, so a collector that quietly returns less
# than it should takes its tests with it: the suite reports a smaller number and stays green. These
# assert the collectors reach the corners of the package that are easy to lose.


def test_public_objects_reaches_every_layer_of_the_package():
    found = {name for name, _ in public_objects()}

    assert "pytensor_ml.model.Model" in found, "a class re-exported from the package root"
    assert "pytensor_ml.optim.alias.adam" in found, "a function nested two packages deep"
    assert "pytensor_ml.layers.linear" in found, "a module, not just the objects inside it"


def test_public_objects_skips_the_backend_dispatch_modules():
    dispatch_modules = [name for name, _ in public_objects() if ".dispatch." in name]

    # Importing one pulls in the backend itself, and the core test jobs install none of them.
    assert not dispatch_modules


def test_attribute_docstrings_finds_a_type_alias():
    documented = dict(attribute_docstrings(pytensor_ml.optim.base))

    # A type alias cannot carry __doc__, so the literal beneath the assignment is its only docstring
    # and the only thing autodoc renders.
    assert "Examples" in documented["pytensor_ml.optim.base.Transform"]


def test_all_docstrings_covers_both_kinds_of_docstring():
    documented = dict(all_docstrings())

    assert "pytensor_ml.model.Model" in documented, "an object's own __doc__"
    assert "pytensor_ml.optim.base.Schedule" in documented, "a module attribute's docstring"
