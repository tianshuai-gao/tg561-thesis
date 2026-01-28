def test_sql_sanity_placeholder():
    # Placeholder sanity test only: NOT PICARD.
    # We keep it only to ensure our pipeline runs.
    good = "SELECT * FROM author;"
    bad = "SELECT FROM ;"
    assert "SELECT" in good.upper() and "FROM" in good.upper()
    assert "SELECT" in bad.upper() and "FROM" in bad.upper()

