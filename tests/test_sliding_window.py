import array
from sliding_window import SlidingWindow

# --- minimal test harness ---
_pass = _fail = 0

def check(name, cond):
    global _pass, _fail
    if cond:
        print("  PASS", name)
        _pass += 1
    else:
        print("  FAIL", name)
        _fail += 1

def make_chunk(hop, cols, value):
    return array.array('h', [value] * (hop * cols))


# --- test functions ---

def test_none_until_full():
    sw = SlidingWindow(100, 20, 6)
    for i in range(4):
        check("push {} -> None".format(i+1), sw.push(make_chunk(20, 6, i)) is None)
    mv = sw.push(make_chunk(20, 6, 99))
    check("push 5 -> memoryview", mv is not None)
    check("ready() True", sw.ready())

def test_window_length():
    sw = SlidingWindow(100, 20, 6)
    for _ in range(5):
        mv = sw.push(make_chunk(20, 6, 1))
    check("len == window*cols", len(mv) == 100 * 6)

def test_data_ordering():
    # window=4 rows, hop=2, 1-col => 2 pushes to fill
    sw = SlidingWindow(4, 2, 1)
    for v in (10, 20):
        mv = sw.push(array.array('h', [v, v]))
    check("row 0,1 = oldest (10)", mv[0] == 10 and mv[1] == 10)
    check("row 2,3 = newest (20)", mv[2] == 20 and mv[3] == 20)

def test_sliding_evicts_oldest():
    sw = SlidingWindow(4, 2, 1)
    for v in (10, 20):
        sw.push(array.array('h', [v, v]))
    mv = sw.push(array.array('h', [30, 30]))
    check("oldest now 20", mv[0] == 20 and mv[1] == 20)
    check("newest now 30", mv[2] == 30 and mv[3] == 30)

def test_element_type():
    sw = SlidingWindow(100, 20, 6)
    for _ in range(5):
        mv = sw.push(make_chunk(20, 6, 7))
    check("element is int", type(mv[0]) is int)

def test_int16_boundaries():
    sw = SlidingWindow(20, 20, 1)
    mv = sw.push(array.array('h', [32767, -32768] * 20))
    check("max int16 stored", mv[0] == 32767)
    check("min int16 stored", mv[1] == -32768)

def test_3_columns():
    sw = SlidingWindow(10, 5, 3)
    for _ in range(2):
        mv = sw.push(make_chunk(5, 3, 42))
    check("3-col len", len(mv) == 10 * 3)
    check("3-col value", mv[0] == 42)

def test_reset():
    sw = SlidingWindow(100, 20, 6)
    for _ in range(5):
        sw.push(make_chunk(20, 6, 9))
    sw.reset()
    check("ready() False after reset", not sw.ready())
    check("first push after reset -> None", sw.push(make_chunk(20, 6, 1)) is None)
    check("buffer zeroed", sw._buf[0] == 0)

def test_no_allocation_on_push():
    sw = SlidingWindow(100, 20, 6)
    for _ in range(5):
        mv = sw.push(make_chunk(20, 6, 1))
    mv2 = sw.push(make_chunk(20, 6, 2))
    check("same memoryview id", id(mv) == id(mv2))


# --- runner ---

tests = [
    ("returns None until full",       test_none_until_full),
    ("correct window length",          test_window_length),
    ("data ordering",                  test_data_ordering),
    ("sliding evicts oldest",          test_sliding_evicts_oldest),
    ("element type is int",            test_element_type),
    ("int16 boundary values",          test_int16_boundaries),
    ("3-column variant",               test_3_columns),
    ("reset clears state",             test_reset),
    ("no allocation on push",          test_no_allocation_on_push),
]

for i, (label, fn) in enumerate(tests, 1):
    print("\n[ {} ] {}".format(i, label))
    fn()

print("\n{} passed, {} failed".format(_pass, _fail))
if _fail:
    raise SystemExit(1)
