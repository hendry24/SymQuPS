from symqups import s
from test_object_manipulation import arithmetic_test

from utils import expected_to_fail

from symqups._internal.cache import sub_cache
sub_cache.clear()

def test_s_parameter():
    arithmetic_test(s.val)
    expected_to_fail(lambda: arithmetic_test(s))