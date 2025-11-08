"""Tests for rubiks_cube.app module-level ROUTES constant and get_router function."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch


def test_routes_constant_exists() -> None:
    """Test that ROUTES constant exists at module level."""
    # Mock dependencies before importing
    with (
        patch("rubiks_cube.app.stx") as mock_stx,
        patch("rubiks_cube.app.st") as mock_st,
    ):
        # Setup minimal mocks
        mock_stx.CookieManager.return_value = MagicMock()
        mock_st.session_state = {}
        mock_st.set_page_config = MagicMock()

        # Import the module
        from rubiks_cube import app

        # Verify ROUTES exists
        assert hasattr(app, "ROUTES")
        assert isinstance(app.ROUTES, dict)


def test_routes_keys() -> None:
    """Test that ROUTES has the expected keys."""
    # Mock dependencies before importing
    with (
        patch("rubiks_cube.app.stx") as mock_stx,
        patch("rubiks_cube.app.st") as mock_st,
    ):
        # Setup minimal mocks
        mock_stx.CookieManager.return_value = MagicMock()
        mock_st.session_state = {}
        mock_st.set_page_config = MagicMock()

        # Import the module
        from rubiks_cube import app

        # Verify ROUTES keys
        expected_keys = {"autotagger", "solver", "beam-search", "docs"}
        assert set(app.ROUTES.keys()) == expected_keys


def test_get_router_callable() -> None:
    """Test that get_router can be called without raising an exception."""
    # Mock dependencies before importing
    with (
        patch("rubiks_cube.app.stx") as mock_stx,
        patch("rubiks_cube.app.st") as mock_st,
    ):
        # Setup minimal mocks
        mock_stx.CookieManager.return_value = MagicMock()
        mock_st.session_state = {}
        mock_st.set_page_config = MagicMock()
        mock_st.fragment = lambda f: f  # Mock decorator to return function as-is
        mock_stx.Router = MagicMock(return_value=MagicMock())

        # Import the module
        from rubiks_cube import app

        # Call get_router and verify it doesn't raise an exception
        router = app.get_router()
        assert router is not None
        # Verify Router was called with ROUTES
        mock_stx.Router.assert_called_once_with(app.ROUTES)
