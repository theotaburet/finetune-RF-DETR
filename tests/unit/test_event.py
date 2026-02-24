"""Tests for AudioEvent and EventList (serialization, filtering, round-trip)."""

from __future__ import annotations

from pathlib import Path

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList


class TestAudioEventRoundTrip:
    """Tests for AudioEvent.to_dict() / from_dict() round-trip."""

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
        # duration_ms included in to_dict output
        assert d["duration_ms"] == 400.0


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

    def test_filter_by_duration_min_and_max(self) -> None:
        events = self._make_events()
        filtered = events.filter_by_duration(min_duration_ms=100.0, max_duration_ms=500.0)
        assert len(filtered) == 1
        assert filtered[0].class_id == 1

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
