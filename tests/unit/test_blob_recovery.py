def test_import():
    import openptv2.blob_recovery as br
    assert hasattr(br, "recover")
