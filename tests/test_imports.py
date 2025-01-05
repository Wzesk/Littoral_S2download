"""Test module imports and basic functionality."""
from pathlib import Path

from littoral import ee_s2, littoral_sites, tario


def test_ee_s2_imports():
    """Test all main functions from ee_s2 are available."""
    assert hasattr(ee_s2, "connect")
    assert hasattr(ee_s2, "get_image_collection")
    assert hasattr(ee_s2, "retrieve_rgb_nir_from_collection")
    assert hasattr(ee_s2, "process_collection_images")


def test_tario_functionality(tmp_path: Path):
    """Test TAR archive functionality."""
    tar = tario.tar_io(str(tmp_path / "test.tar"))
    assert hasattr(tar, "save_to_tar")
    assert hasattr(tar, "get_from_tar")
    assert hasattr(tar, "get_tar_filenames")


def test_littoral_sites_imports():
    """Test site management functions are available."""
    assert hasattr(littoral_sites, "load_sites")
    assert hasattr(littoral_sites, "get_site_by_name")
    assert hasattr(littoral_sites, "set_last_run")
