from __future__ import annotations

from depthsync.depth_visualization import select_inspection_frames


def test_scene_03_inspection_includes_worst_jump_neighbors() -> None:
    frames = select_inspection_frames(
        "03",
        frame_count=90,
        anchor_index=45,
        worst_jump_frame=61,
        configured=None,
    )

    assert frames == [45, 60, 61, 62]


def test_configured_inspection_frames_are_clipped_and_deduplicated() -> None:
    frames = select_inspection_frames(
        "01",
        frame_count=5,
        anchor_index=2,
        worst_jump_frame=3,
        configured=[0, 2, 2, 8],
    )

    assert frames == [0, 2, 4]
