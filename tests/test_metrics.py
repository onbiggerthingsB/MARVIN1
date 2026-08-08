from server.metrics import TurnLog


def test_percentiles_over_recorded_turns():
    log = TurnLog(window=100)
    for ms in (100, 200, 300, 400, 1000):
        t_release = 1000.0
        log.record_utterance(t_release=t_release, t_utterance=t_release + ms / 1000)
        log.record_first_audio(t_first_audio=t_release + ms / 1000 + 0.5)
    s = log.summary()
    assert s["turns"] == 5
    assert s["release_to_final_p50"] == 300
    assert s["release_to_final_p95"] == 1000
    assert s["final_to_audio_p50"] == 500


def test_missing_timestamps_are_skipped():
    log = TurnLog()
    log.record_utterance(t_release=None, t_utterance=5.0)
    log.record_first_audio(t_first_audio=None)
    assert log.summary()["turns"] == 0
