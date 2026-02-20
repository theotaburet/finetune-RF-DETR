"""Tests for AudioEvent and EventList (serialization, filtering, round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList


class TestAudioEventToFromDict:
    """Tests for AudioEvent.to_dict() and from_dict() round-trip."""

    def test_round_trip_minimal(self) -> None:
        event = AudioEvent(start_ms=100.0, end_ms=500.0, class_id=0)
        d = event.to_dict()
        restored = AudioEvent.from_dict(d)
        assert restored.start_ms == event.start_ms
        assert restored.end_ms == event.end_ms
        assert restored.class_id == event.class_id
        assert restored.class_name is None
        assert restored.score == 1.0

    def test_round_trip_full(self) -> None:
        event = AudioEvent(
            start_ms=100.0,
            end_ms=500.0,
            class_id=2,
            class_name="whale",
            score=0.95,
            min_freq_hz=200.0,
            max_freq_hz=5000.0,
            source_windows=[0, 1, 2],
            metadata={"detector": "rfdetr"},
        )
        d = event.to_dict()
        restored = AudioEvent.from_dict(d)
        assert restored.start_ms == 100.0
        assert restored.end_ms == 500.0
        assert restored.class_id == 2
        assert restored.class_name == "whale"
        assert restored.score == 0.95
        assert restored.min_freq_hz == 200.0
        assert restored.max_freq_hz == 5000.0
        assert restored.source_windows == [0, 1, 2]
        assert restored.metadata == {"detector": "rfdetr"}

    def test_to_dict_includes_duration(self) -> None:
        event = AudioEvent(start_ms=100.0, end_ms=500.0, class_id=0)
        d = event.to_dict()
        assert d["duration_ms"] == 400.0

    def test_from_dict_missing_optional_fields(self) -> None:
        d = {"start_ms": 0.0, "end_ms": 100.0, "class_id": 1}
        event = AudioEvent.from_dict(d)
        assert event.class_name is None
        assert event.score == 1.0
        assert event.min_freq_hz is None
        assert event.max_freq_hz is None
        assert event.source_windows == []
        assert event.metadata == {}

    def test_from_dict_with_extra_fields_ignored(self) -> None:
        """from_dict should not crash on extra keys (e.g. duration_ms from to_dict)."""
        d = {
            "start_ms": 0.0,
            "end_ms": 100.0,
            "class_id": 0,
            "duration_ms": 100.0,
            "extra_key": "ignored",
        }
        event = AudioEvent.from_dict(d)
        assert event.start_ms == 0.0


class TestAudioEventProperties:
    """Tests for AudioEvent computed properties."""

    def test_duration_ms(self) -> None:
        event = AudioEvent(start_ms=100.0, end_ms=500.0, class_id=0)
        assert event.duration_ms == 400.0

    def test_duration_s(self) -> None:
        event = AudioEvent(start_ms=0.0, end_ms=1000.0, class_id=0)
        assert event.duration_s == 1.0

    def test_center_ms(self) -> None:
        event = AudioEvent(start_ms=100.0, end_ms=500.0, class_id=0)
        assert event.center_ms == 300.0


class TestEventListSaveLoad:
    """Tests for EventList.save() and EventList.load() round-trip."""

    def _make_event_list(self) -> EventList:
        return EventList(
            events=[
                AudioEvent(start_ms=0.0, end_ms=100.0, class_id=0, class_name="bird", score=0.9),
                AudioEvent(start_ms=200.0, end_ms=500.0, class_id=1, class_name="frog", score=0.8),
            ],
            audio_path="/data/test.wav",
            duration_ms=1000.0,
            class_names={0: "bird", 1: "frog"},
        )

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        original = self._make_event_list()
        save_path = tmp_path / "events.json"
        original.save(save_path)
        loaded = EventList.load(save_path)

        assert len(loaded) == len(original)
        assert loaded.audio_path == str(original.audio_path)
        assert loaded.duration_ms == original.duration_ms
        assert loaded.class_names == original.class_names

        for orig, rest in zip(original.events, loaded.events):
            assert rest.start_ms == orig.start_ms
            assert rest.end_ms == orig.end_ms
            assert rest.class_id == orig.class_id
            assert rest.class_name == orig.class_name
            assert rest.score == orig.score

    def test_class_names_int_keys_preserved(self, tmp_path: Path) -> None:
        """The critical fix: int keys must survive JSON round-trip."""
        original = EventList(class_names={0: "bird", 1: "frog", 42: "whale"})
        save_path = tmp_path / "events.json"
        original.save(save_path)
        loaded = EventList.load(save_path)

        assert all(isinstance(k, int) for k in loaded.class_names)
        assert loaded.class_names == {0: "bird", 1: "frog", 42: "whale"}

    def test_class_names_keys_are_strings_in_json(self, tmp_path: Path) -> None:
        """JSON serialization converts int keys to strings; load() must fix this."""
        save_path = tmp_path / "events.json"
        data = {
            "events": [],
            "class_names": {"0": "bird", "1": "frog"},
            "duration_ms": 0.0,
        }
        save_path.write_text(json.dumps(data))
        loaded = EventList.load(save_path)
        assert loaded.class_names == {0: "bird", 1: "frog"}
        assert all(isinstance(k, int) for k in loaded.class_names)

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        event_list = self._make_event_list()
        deep_path = tmp_path / "a" / "b" / "c" / "events.json"
        event_list.save(deep_path)
        assert deep_path.exists()

    def test_load_missing_events_key(self, tmp_path: Path) -> None:
        save_path = tmp_path / "events.json"
        save_path.write_text(json.dumps({"duration_ms": 500.0}))
        loaded = EventList.load(save_path)
        assert len(loaded) == 0
        assert loaded.duration_ms == 500.0

    def test_load_missing_class_names_key(self, tmp_path: Path) -> None:
        save_path = tmp_path / "events.json"
        save_path.write_text(json.dumps({"events": []}))
        loaded = EventList.load(save_path)
        assert loaded.class_names == {}

    def test_load_empty_event_list(self, tmp_path: Path) -> None:
        event_list = EventList()
        save_path = tmp_path / "empty.json"
        event_list.save(save_path)
        loaded = EventList.load(save_path)
        assert len(loaded) == 0
        assert loaded.class_names == {}


class TestEventListFiltering:
    """Tests for EventList filtering methods."""

    def _make_events(self) -> EventList:
        return EventList(
            events=[
                AudioEvent(start_ms=0.0, end_ms=50.0, class_id=0, score=0.9),
                AudioEvent(start_ms=100.0, end_ms=500.0, class_id=1, score=0.5),
                AudioEvent(start_ms=600.0, end_ms=2000.0, class_id=0, score=0.3),
            ],
            class_names={0: "bird", 1: "frog"},
        )

    def test_filter_by_duration_min(self) -> None:
        events = self._make_events()
        filtered = events.filter_by_duration(min_duration_ms=100.0)
        assert len(filtered) == 2
        assert all(e.duration_ms >= 100.0 for e in filtered)

    def test_filter_by_duration_max(self) -> None:
        events = self._make_events()
        filtered = events.filter_by_duration(max_duration_ms=500.0)
        assert len(filtered) == 2
        assert all(e.duration_ms <= 500.0 for e in filtered)

    def test_filter_by_duration_min_and_max(self) -> None:
        events = self._make_events()
        filtered = events.filter_by_duration(min_duration_ms=100.0, max_duration_ms=500.0)
        assert len(filtered) == 1
        assert filtered[0].class_id == 1

    def test_filter_preserves_class_names(self) -> None:
        events = self._make_events()
        filtered = events.filter_by_duration(min_duration_ms=100.0)
        assert filtered.class_names == events.class_names

    def test_get_class_counts(self) -> None:
        events = self._make_events()
        counts = events.get_class_counts()
        assert counts == {0: 2, 1: 1}

    def test_sort_by_time(self) -> None:
        events = EventList(
            events=[
                AudioEvent(start_ms=500.0, end_ms=600.0, class_id=0),
                AudioEvent(start_ms=100.0, end_ms=200.0, class_id=0),
                AudioEvent(start_ms=300.0, end_ms=400.0, class_id=0),
            ]
        )
        sorted_events = events.sort_by_time()
        assert [e.start_ms for e in sorted_events] == [100.0, 300.0, 500.0]

    def test_sort_by_score(self) -> None:
        events = self._make_events()
        sorted_events = events.sort_by_score(descending=True)
        scores = [e.score for e in sorted_events]
        assert scores == sorted(scores, reverse=True)
