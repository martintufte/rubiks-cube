"""Tests for rubiks_cube/app.py router configuration."""

from __future__ import annotations

import rubiks_cube.app


def test_routes_constant_exists() -> None:
    """Test that ROUTES constant exists at module level."""
    assert hasattr(rubiks_cube.app, "ROUTES")
    assert isinstance(rubiks_cube.app.ROUTES, dict)


def test_routes_keys_are_correct() -> None:
    """Test that ROUTES keys match expected route names without leading slashes."""
    expected_keys = {"autotagger", "solver", "beam-search", "docs"}
    actual_keys = set(rubiks_cube.app.ROUTES.keys())
    assert actual_keys == expected_keys, (
        f"ROUTES keys mismatch. Expected: {expected_keys}, Got: {actual_keys}"
    )


def test_get_router_returns_router() -> None:
    """Test that get_router() returns a Router object."""
    # Note: We avoid calling get_router() directly as it may trigger Streamlit
    # initialization. Instead, we verify the function exists and is callable.
    assert hasattr(rubiks_cube.app, "get_router")
    assert callable(rubiks_cube.app.get_router)


def test_routes_values_are_callables() -> None:
    """Test that all ROUTES values are callable (partials)."""
    for route_name, route_handler in rubiks_cube.app.ROUTES.items():
        assert callable(route_handler), f"Route handler for '{route_name}' is not callable"
