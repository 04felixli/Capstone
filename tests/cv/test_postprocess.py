import pytest

from haptos.config import (
    CENTER_REGION,
    COMMAND_FORWARD,
    COMMAND_GO_LEFT,
    COMMAND_GO_RIGHT,
    COMMAND_STOP,
    LEFT_REGION,
    RIGHT_REGION,
)
from haptos.cv.postprocess import (
    filter_and_enrich_detections,
    generate_navigation_hint,
    map_bbox_to_region,
)
from haptos.types import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw(class_name="person", confidence=0.9, bbox=(260, 0, 380, 100)):
    """Raw Detection as the detector emits it — no region or obstacle info."""
    return Detection(class_name=class_name, confidence=confidence, bbox=bbox)


def _obstacle(region):
    return Detection(
        class_name="person", confidence=0.9,
        bbox=(0, 0, 10, 10), region=region, is_obstacle=True,
    )


def _non_obstacle(region):
    return Detection(
        class_name="laptop", confidence=0.9,
        bbox=(0, 0, 10, 10), region=region, is_obstacle=False,
    )


# ---------------------------------------------------------------------------
# map_bbox_to_region
# frame_width=300 → one_third=100, two_thirds=200 (clean boundaries)
# ---------------------------------------------------------------------------

class TestMapBboxToRegion:
    def test_center_left_of_one_third(self):
        # center_x=50 → LEFT
        assert map_bbox_to_region((0, 0, 100, 100), frame_width=300) == LEFT_REGION

    def test_center_in_middle_third(self):
        # center_x=150 → CENTER
        assert map_bbox_to_region((100, 0, 200, 100), frame_width=300) == CENTER_REGION

    def test_center_right_of_two_thirds(self):
        # center_x=250 → RIGHT
        assert map_bbox_to_region((200, 0, 300, 100), frame_width=300) == RIGHT_REGION

    def test_boundary_at_one_third_goes_center(self):
        # center_x=100 exactly: not < one_third → CENTER
        assert map_bbox_to_region((100, 0, 100, 0), frame_width=300) == CENTER_REGION

    def test_boundary_at_two_thirds_goes_right(self):
        # center_x=200 exactly: not < two_thirds → RIGHT
        assert map_bbox_to_region((200, 0, 200, 0), frame_width=300) == RIGHT_REGION


# ---------------------------------------------------------------------------
# filter_and_enrich_detections
# ---------------------------------------------------------------------------

class TestFilterAndEnrichDetections:
    def test_filters_below_threshold(self):
        result = filter_and_enrich_detections(
            [_raw(confidence=0.3)], frame_width=640, confidence_threshold=0.4
        )
        assert result == []

    def test_passes_at_exact_threshold(self):
        # filter uses strict <, so confidence == threshold passes
        result = filter_and_enrich_detections(
            [_raw(confidence=0.4)], frame_width=640, confidence_threshold=0.4
        )
        assert len(result) == 1

    def test_assigns_correct_region(self):
        # bbox center at 320 on 640px frame → CENTER
        result = filter_and_enrich_detections(
            [_raw(bbox=(260, 0, 380, 100))], frame_width=640, confidence_threshold=0.4
        )
        assert result[0].region == CENTER_REGION

    def test_marks_obstacle_class(self):
        result = filter_and_enrich_detections(
            [_raw(class_name="person")], frame_width=640, confidence_threshold=0.4
        )
        assert result[0].is_obstacle is True

    def test_obstacle_matching_is_case_insensitive(self):
        result = filter_and_enrich_detections(
            [_raw(class_name="Person")], frame_width=640, confidence_threshold=0.4
        )
        assert result[0].is_obstacle is True

    def test_non_obstacle_class_not_flagged(self):
        result = filter_and_enrich_detections(
            [_raw(class_name="laptop")], frame_width=640, confidence_threshold=0.4
        )
        assert result[0].is_obstacle is False

    def test_raw_detection_has_no_region_before_enrichment(self):
        raw = _raw()
        assert raw.region is None
        assert raw.is_obstacle is False

    def test_preserves_class_and_confidence(self):
        result = filter_and_enrich_detections(
            [_raw(class_name="car", confidence=0.85)], frame_width=640, confidence_threshold=0.4
        )
        assert result[0].class_name == "car"
        assert result[0].confidence == 0.85

    def test_empty_input(self):
        assert filter_and_enrich_detections([], frame_width=640, confidence_threshold=0.4) == []


# ---------------------------------------------------------------------------
# generate_navigation_hint
# ---------------------------------------------------------------------------

class TestGenerateNavigationHint:
    def test_no_detections(self):
        assert generate_navigation_hint([]) == COMMAND_FORWARD

    def test_non_obstacle_detections_only(self):
        assert generate_navigation_hint([_non_obstacle(CENTER_REGION)]) == COMMAND_FORWARD

    def test_center_obstacle_stops(self):
        assert generate_navigation_hint([_obstacle(CENTER_REGION)]) == COMMAND_STOP

    def test_left_only_goes_right(self):
        assert generate_navigation_hint([_obstacle(LEFT_REGION)]) == COMMAND_GO_RIGHT

    def test_right_only_goes_left(self):
        assert generate_navigation_hint([_obstacle(RIGHT_REGION)]) == COMMAND_GO_LEFT

    def test_left_and_right_stops(self):
        assert generate_navigation_hint(
            [_obstacle(LEFT_REGION), _obstacle(RIGHT_REGION)]
        ) == COMMAND_STOP

    def test_left_and_center_stops(self):
        assert generate_navigation_hint(
            [_obstacle(LEFT_REGION), _obstacle(CENTER_REGION)]
        ) == COMMAND_STOP

    def test_mixed_obstacle_and_non_obstacle_left(self):
        # non-obstacle in center should not trigger STOP
        assert generate_navigation_hint(
            [_obstacle(LEFT_REGION), _non_obstacle(CENTER_REGION)]
        ) == COMMAND_GO_RIGHT

    def test_obstacle_with_no_region_ignored(self):
        # raw Detection before enrichment has region=None — must not crash or count
        raw = Detection(
            class_name="person", confidence=0.9,
            bbox=(0, 0, 10, 10), is_obstacle=True,
        )
        assert generate_navigation_hint([raw]) == COMMAND_FORWARD
