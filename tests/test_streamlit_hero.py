from __future__ import annotations

import base64

import streamlit_app


def test_hero_background_image_is_bundled_as_jpeg_data_uri() -> None:
    assert streamlit_app._HERO_IMAGE_PATH.is_file()

    prefix, encoded = streamlit_app._hero_image_data_uri().split(",", maxsplit=1)

    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded).startswith(b"\xff\xd8\xff")


def test_hero_renders_supplied_image_behind_existing_content(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(streamlit_app.st, "markdown", fake_markdown)

    streamlit_app._render_model_hero()

    body = str(captured["body"])
    assert 'class="designer-hero-image"' in body
    assert f'src="{streamlit_app._hero_image_data_uri()}"' in body
    assert body.index('class="designer-hero-image"') < body.index(
        'class="designer-hero-content"'
    )
    assert "Goat Farm Financial Model" in body
    assert captured["unsafe_allow_html"] is True