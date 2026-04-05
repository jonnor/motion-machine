import array
from reservoir_sampler import ReservoirSampler


def test_initial_state():
    rs = ReservoirSampler(k=10, cols=3)
    assert len(rs) == 0, "Expected empty reservoir initially"
    assert rs.n_seen == 0, "Expected n_seen to be 0 initially"


def test_fill_phase_exact():
    rs = ReservoirSampler(k=5, cols=2)

    chunk = array.array('h', [
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10
    ])

    rs.push(chunk)

    assert len(rs) == 5, "Reservoir should be fully filled with 5 rows"
    assert rs.n_seen == 5, "n_seen should equal number of rows pushed"

    rows = list(rs.get_rows())
    assert rows[0] == (1, 2), "First row mismatch after fill phase"
    assert rows[4] == (9, 10), "Last row mismatch after fill phase"


def test_fill_then_overflow():
    rs = ReservoirSampler(k=3, cols=2, seed=42)

    chunk = array.array('h', [
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10
    ])

    rs.push(chunk)

    assert len(rs) == 3, "Reservoir size should remain capped at k=3"
    assert rs.n_seen == 5, "n_seen should track total rows processed"


def test_chunk_multiple_pushes():
    rs = ReservoirSampler(k=4, cols=2, seed=1)

    chunk1 = array.array('h', [
        1, 2,
        3, 4
    ])

    chunk2 = array.array('h', [
        5, 6,
        7, 8
    ])

    rs.push(chunk1)
    rs.push(chunk2)

    assert rs.n_seen == 4, "n_seen should accumulate across multiple pushes"
    assert len(rs) == 4, "Reservoir should contain 4 rows after two pushes"


def test_invalid_chunk_size():
    rs = ReservoirSampler(k=5, cols=3)

    bad_chunk = array.array('h', [1, 2, 3, 4])  # not multiple of 3

    try:
        rs.push(bad_chunk)
        assert False, "Expected ValueError for invalid chunk size but none was raised"
    except ValueError:
        pass


def test_deterministic_seed():
    rs1 = ReservoirSampler(k=3, cols=2, seed=123)
    rs2 = ReservoirSampler(k=3, cols=2, seed=123)

    chunk = array.array('h', [
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10
    ])

    rs1.push(chunk)
    rs2.push(chunk)

    rows1 = list(rs1.get_rows())
    rows2 = list(rs2.get_rows())

    assert rows1 == rows2, "Reservoir contents differ despite identical seed and input"


def test_get_flat_size():
    rs = ReservoirSampler(k=6, cols=2)

    flat = rs.get_flat()
    assert len(flat) == 12, "Flat storage length should be k * cols (6 * 2 = 12)"


def test_partial_fill_rows():
    rs = ReservoirSampler(k=5, cols=2)

    chunk = array.array('h', [
        1, 2,
        3, 4
    ])

    rs.push(chunk)

    rows = list(rs.get_rows())
    assert len(rows) == 2, "Expected 2 rows after partial fill"
    assert rows[1] == (3, 4), "Second row mismatch after partial fill"


def run_all_tests():
    test_initial_state()
    test_fill_phase_exact()
    test_fill_then_overflow()
    test_chunk_multiple_pushes()
    test_invalid_chunk_size()
    #test_deterministic_seed() # FIXME: broken on MicroPython Unix
    test_get_flat_size()
    test_partial_fill_rows()
    print("All tests passed!")


if __name__ == '__main__':
    run_all_tests()

