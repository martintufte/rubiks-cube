import os
from functools import partial
from typing import Any
from typing import Final

import extra_streamlit_components as stx
import streamlit as st

from rubiks_cube.configuration.path_definitions import RESOURCES_DIR
from rubiks_cube.pages import app
from rubiks_cube.pages import docs
from rubiks_cube.pages import solver
from rubiks_cube.utils.parsing import parse_scramble
from rubiks_cube.utils.parsing import parse_user_input

st.set_page_config(
    page_title="Fewest Moves Engine",
    page_icon=os.path.join(RESOURCES_DIR, "favicon.png"),
)


@st.fragment
def get_cookie_manager() -> stx.CookieManager:
    """Get the cookie manager.

    Returns:
        stx.CookieManager: Cookie manager.
    """
    return stx.CookieManager()


COOKIE_MANAGER: Final[stx.CookieManager] = get_cookie_manager()
DEFAULT_SESSION: Final[dict[str, Any]] = {
    "scramble": parse_scramble(COOKIE_MANAGER.get("scramble_input") or ""),
    "user": parse_user_input(COOKIE_MANAGER.get("user_input") or ""),
}
for key, default in DEFAULT_SESSION.items():
    if key not in st.session_state:
        setattr(st.session_state, key, default)


@st.fragment
def get_router() -> stx.Router:
    """Return the router for the app.

    Returns:
        stx.Router: The router.
    """
    return stx.Router(
        {
            "/app": partial(
                app,
                session=st.session_state,
                cookie_manager=COOKIE_MANAGER,
            ),
            "/solver": partial(
                solver,
                session=st.session_state,
                cookie_manager=COOKIE_MANAGER,
            ),
            "/docs": partial(
                docs,
                session=st.session_state,
                cookie_manager=COOKIE_MANAGER,
            ),
        }
    )


def router() -> None:
    """Page footer with navigation."""
    ROUTER: stx.Router = get_router()
    ROUTER.show_route_view()

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    if "initialized" not in st.session_state:
        st.session_state.__setattr__("initialized", True)
        ROUTER.route("app")

    cols = st.columns([1, 1, 1])

    with cols[0]:
        if st.button(":blue[APP]", key="app"):
            ROUTER.route("app")
    with cols[1]:
        if st.button(":blue[SOLVER]", key="solver"):
            ROUTER.route("solver")
    with cols[2]:
        if st.button(":blue[DOCS]", key="docs"):
            ROUTER.route("docs")


if __name__ == "__main__":
    router()
