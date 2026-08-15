from pathlib import Path


DEMO = Path(__file__).resolve().parents[1] / "demo" / "index.html"


def test_shadow_and_wuji_cards_are_swapped_and_new_wuji_is_published() -> None:
    html = DEMO.read_text()
    shadow_source = (
        '<source src="showcase/five-finger-shadow-arm-background-locked.mp4"'
    )
    wuji_source = (
        '<source src="showcase/'
        'human-to-wuji-real-hardware-appearance-comparison-20p7s.mp4"'
    )
    assert html.index(shadow_source) < html.index(wuji_source)
    assert "human-to-wuji-hand-source-scene-locked-comparison-20p7s.mp4" not in html
    assert "human-to-wuji-hand-shadow-style-scene-locked-comparison-20p7s.mp4" not in html
    assert "wuji-real-hardware-reference-manifest.json" in html
