from symqups import s
from test_object_manipulation import arithmetic_test

def test_s_parameter():
    arithmetic_test(s.val)
    try:
        arithmetic_test(s)
        raise RuntimeError("Test failed.")
    except:
        pass