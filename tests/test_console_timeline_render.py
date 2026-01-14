from rich.console import Console

from kiva.console import KivaLiveRenderer


def test_rich_timeline_render_is_deterministic():
    renderer = KivaLiveRenderer("p")
    status = {
        "scope": "worker",
        "state": "verifying",
        "timeline": [
            {
                "scope": "worker",
                "state": "initializing",
                "timestamp": 1000.0,
                "message": "init",
            },
            {
                "scope": "worker",
                "state": "preprocessing",
                "timestamp": 1000.01,
                "message": "prep",
            },
            {
                "scope": "worker",
                "state": "verifying",
                "timestamp": 1000.02,
                "message": "verify",
            },
        ],
    }
    renderer.on_verification_state_changed(status)

    panel = renderer._render_verification_timeline_panel(limit=0)
    console = Console(record=True, width=100)
    console.print(panel)
    text = console.export_text()

    assert "Verification Timeline" in text
    assert "worker" in text
    assert "initializing" in text
    assert "preprocessing" in text
    assert "verifying" in text
    assert "0.00s" in text
