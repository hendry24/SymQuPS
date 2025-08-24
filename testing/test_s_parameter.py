from symqups import s
from test_object_manipulation import arithmetic

def test_s_parameter():
    arithmetic(s.val)
    try:
        arithmetic(s)
        raise RuntimeError("Test failed.")
    except:
        pass