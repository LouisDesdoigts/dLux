import sys
import os

platform: str = sys.platform
not_lin_err: str = """
    The installation test is designed to run on a linux machine. In particular,
    this test is designed to be run by github actions.
"""

if platform not in ["linux", "linux2"]:
    print(f"Error: {not_lin_err}")


def test_install_dLux():
    install_ok = os.system("pip install --quiet .")

    assert install_ok == 0

    os.system("rm -r dLux.egg-info")
    os.system("rm -r build")


def test_import_dLux():
    import dLux

    dLux.optics
    assert True
