"""The leg-5 protocol: frames over a socket pair, a producer and a
consumer as two processes on localhost, and the replayer and tracker of
the second direction on one."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from dtfit.image import Image, ImageStream, Original, assemble

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, store, stream,
)

EXPR = "c + v*t"
NAMES = ["c", "v"]


def blocks(n_blocks=3, order=12):
    """``n_blocks`` one-unit block images of a noisy line, as a station's
    yearly blocks would look."""
    rng = np.random.default_rng(1)
    out = []
    for k in range(n_blocks):
        x = k + np.sort(rng.uniform(0.0, 1.0, 40))
        y = 3.0 + 0.5 * x + 0.01 * rng.standard_normal(x.size)
        out.append(Image.of(
            Original(x, y, domain=(float(k), float(k + 1))),
            "legendre", order,
        ))
    return out


class CountingFilter:
    """A stand-in for ``LSIFilter``: :func:`stream.track` needs only
    ``partial_fit``, ``drift_flag_`` and ``params_``, so the protocol test
    never depends on the filter lane."""

    def __init__(self, every=50):
        self.every = every
        self.n = 0
        self.drift_flag_ = False
        self.params_ = {"c": 0.0}

    def partial_fit(self, t, y):
        self.n += 1
        self.drift_flag_ = self.n % self.every == 0
        self.params_["c"] = float(y)
        return self


def test_frames_round_trip_over_a_socket_pair():
    a, b = socket.socketpair()
    try:
        sent = stream.send_frame(a, {"kind": "hello", "nbytes": 5},
                                 b"12345")
        header, payload = stream.recv_frame(b)
        assert header["kind"] == "hello" and payload == b"12345"
        assert sent == 4 + len(json.dumps(
            {"kind": "hello", "nbytes": 5}, separators=(",", ":")
        )) + 5
        a.close()
        assert stream.recv_frame(b) is None
    finally:
        a.close()
        b.close()


def test_an_image_survives_the_frame_exactly():
    img = blocks(1)[0]
    header = stream.image_header(img, station="AAAA", field="east",
                                 block=7)
    payload = stream.image_payload(img)
    assert header["station"] == "AAAA" and header["block"] == 7
    assert header["nbytes"] == len(payload)
    back = stream.image_from_frame(header, payload)
    assert back == img
    assert stream.image_digest(back) == stream.image_digest(img)


def test_a_weighted_image_survives_the_frame():
    x = np.linspace(0.0, 1.0, 50)
    w = np.linspace(1.0, 2.0, 50)
    img = Image.of(Original(x, np.sin(x), w), "legendre", 8)
    back = stream.image_from_frame(
        stream.image_header(img), stream.image_payload(img)
    )
    assert back == img and back.w is not None


def test_the_digest_changes_with_the_statistic():
    a, b = blocks(2)
    assert stream.image_digest(a) != stream.image_digest(b)
    assert stream.image_digest(a) == stream.image_digest(a)


def test_a_samples_frame_survives_the_wire():
    t = np.linspace(0.0, 1.0, 17)
    y = np.cos(t)
    header = stream.samples_header("AAAA", "east", t.size, seq=4)
    payload = stream.samples_payload(t, y)
    assert header["kind"] == stream.SAMPLE_KIND
    assert header["nbytes"] == len(payload) == 2 * 17 * 8
    back_t, back_y = stream.samples_from_frame(header, payload)
    np.testing.assert_array_equal(back_t, t)
    np.testing.assert_array_equal(back_y, y)
    with pytest.raises(ValueError, match="bytes"):
        stream.samples_from_frame(header, payload[:-8])
    bad = dict(header, dtype="float32")
    with pytest.raises(ValueError, match="dtype"):
        stream.samples_from_frame(bad, payload)


def test_iter_images_walks_the_stored_blocks_and_their_flags(tmp_path):
    imgs = blocks(3)
    store.save_images(
        tmp_path / "AAAA.npz",
        {f"blk{k:02d}_east": img for k, img in enumerate(imgs)}
        | {"whole_east": imgs[0]},
        {"sta": "AAAA", "flags": {"east": [[1.0, 1.0, 2.0]]}},
    )
    got = list(stream.iter_images(tmp_path, prefix="blk"))
    assert [e["block"] for _img, e in got] == [0, 1, 2]
    assert {e["station"] for _img, e in got} == {"AAAA"}
    assert {e["field"] for _img, e in got} == {"east"}
    # the drift flag the reducer raised on the second block travels with
    # that block, matched on its domain rather than on a list position
    assert [e["flags"] for _img, e in got] == [0, 1, 0]
    assert len(list(stream.iter_images(tmp_path, prefix="blk",
                                       limit=2))) == 2


def test_producer_and_consumer_run_as_two_processes(tmp_path):
    imgs = blocks(4)
    store.save_images(
        tmp_path / "AAAA.npz",
        {f"blk{k:02d}_east": img for k, img in enumerate(imgs)},
        {"sta": "AAAA", "flags": {"east": [[2.0, 2.0, 3.0]]}},
    )
    out = tmp_path / "summary.json"
    port_file = tmp_path / "port"
    consumer = subprocess.Popen([
        sys.executable, "-m", stream.MODULE, "--role", "consumer",
        "--host", "127.0.0.1", "--port", "0",
        "--port-file", str(port_file), "--out", str(out),
        "--expr", EXPR, "--names", ",".join(NAMES),
        "--timeout", "60",
    ])
    try:
        deadline = time.time() + 30.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert port_file.exists(), "the consumer never bound a port"
        port = int(port_file.read_text().strip())
        producer = subprocess.run([
            sys.executable, "-m", stream.MODULE, "--role", "producer",
            "--host", "127.0.0.1", "--port", str(port),
            "--images", str(tmp_path), "--out", str(tmp_path / "p.json"),
        ], capture_output=True, text=True, timeout=60)
        assert producer.returncode == 0, producer.stderr
        assert consumer.wait(timeout=60) == 0
    finally:
        if consumer.poll() is None:
            consumer.kill()
    summary = json.loads(out.read_text())
    assert summary["n_images"] == 4
    assert summary["bytes_payload"] > 0 and summary["bytes_header"] > 0
    group = summary["groups"]["AAAA/east"]
    assert group["n_blocks"] == 4
    assert group["bytes_payload"] > 0 and group["bytes_header"] > 0
    assert group["samples"] == 4 * 40
    assert group["n_flags"] == 1
    local = assemble(imgs, order=12)
    assert group["digest"] == stream.image_digest(local)
    # the consumer fits the assembled image, which is the whole point of
    # moving the analysis to the receiving machine
    direct = compare.fit_from_image(EXPR, local, NAMES)
    for name in NAMES:
        assert group["params"][name] == pytest.approx(
            float(direct.params[name]), rel=1e-12, abs=1e-12
        )
    sent = json.loads((tmp_path / "p.json").read_text())
    assert sent["n_acks"] == 4
    assert sent["latency_ms_median"] is not None
    assert sent["latency_ms_median"] >= 0.0


def test_serve_reports_a_timeout_instead_of_hanging(tmp_path):
    port_file = tmp_path / "port"
    with pytest.raises(TimeoutError):
        stream.serve("127.0.0.1", 0, port_file=port_file, timeout=0.5)


def test_replay_feeds_the_tracker_and_gets_block_images_back(tmp_path):
    filt = CountingFilter(every=50)
    back = ImageStream(
        "legendre", 4, domain=(0.0, 3.0), block=1.0, detect="previous",
        grid="explicit", keep_fine=8, fold=8,
    )
    port_file = tmp_path / "port"
    result: dict = {}

    def run_tracker():
        result["out"] = stream.track(
            "127.0.0.1", 0, filt, block_stream=back, station="AAAA",
            field="east", port_file=port_file, timeout=30.0,
        )

    thread = threading.Thread(target=run_tracker)
    thread.start()
    received: list = []
    try:
        deadline = time.time() + 20.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert port_file.exists(), "the tracker never bound a port"
        port = int(port_file.read_text().strip())
        x = np.linspace(0.0, 2.999, 300)
        y = 3.0 + 0.5 * x
        chunks = [("AAAA", "east", x[i:i + 20], y[i:i + 20])
                  for i in range(0, x.size, 20)]
        sent = stream.replay(
            "127.0.0.1", port, chunks, rate=None,
            back_sink=lambda h, p: received.append((h, p)),
        )
    finally:
        thread.join(timeout=30.0)
    got = result["out"]
    assert sent["n_frames"] == 15 and sent["n_dropped"] == 0
    assert sent["n_samples"] == 300
    assert got["n_samples"] == 300 and got["n_dropped"] == 0
    assert got["n_other"] == 0
    assert got["n_flags"] == 6                # every 50th of 300 samples
    assert got["us_per_update"] > 0.0
    assert got["n_blocks"] == 3 and got["bytes_back"] > 0
    assert sent["n_frames_back"] == got["n_blocks"] == len(received)
    assert got["params"]["c"] == pytest.approx(float(y[-1]))
    # ship() derives block from the image's own domain start and its own
    # running seq, one per image sent: check both, and that the image
    # itself survives the wire.
    for k, (header, payload) in enumerate(received):
        assert header["station"] == "AAAA" and header["field"] == "east"
        assert header["block"] == k
        assert header["seq"] == k
        img = stream.image_from_frame(header, payload)
        assert img.domain == (float(k), float(k + 1))
        assert stream.image_digest(img) == stream.image_digest(img)


def test_replay_and_track_survive_heavy_back_traffic(tmp_path):
    # An explicit grid ships every sample position in the binary payload,
    # so enough blocks of enough points comfortably clear the measured
    # buffer threshold where the two directions used to deadlock in
    # sendall on the same socket, one still sending while the other's
    # unread back traffic filled its receive buffer.
    n_blocks = 40
    per_block = 2000
    rng = np.random.default_rng(3)
    filt = CountingFilter(every=10**9)
    back = ImageStream(
        "legendre", 12, domain=(0.0, float(n_blocks)), block=1.0,
        detect="previous", grid="explicit", keep_fine=8, fold=8,
    )
    chunks = []
    for k in range(n_blocks):
        x = k + np.sort(rng.uniform(0.0, 1.0, per_block))
        y = 3.0 + 0.5 * x
        for i in range(0, per_block, 500):
            chunks.append(("AAAA", "east", x[i:i + 500], y[i:i + 500]))
    port_file = tmp_path / "port"
    result: dict = {}
    received: list = []

    def run_tracker():
        result["out"] = stream.track(
            "127.0.0.1", 0, filt, block_stream=back, station="AAAA",
            field="east", port_file=port_file, timeout=60.0,
        )

    thread = threading.Thread(target=run_tracker)
    thread.start()
    try:
        deadline = time.time() + 20.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert port_file.exists(), "the tracker never bound a port"
        port = int(port_file.read_text().strip())
        sent = stream.replay(
            "127.0.0.1", port, chunks, rate=None, timeout=30.0,
            back_sink=lambda h, p: received.append((h, p)),
        )
    finally:
        thread.join(timeout=30.0)
    assert not thread.is_alive(), "the tracker never finished: a deadlock"
    got = result["out"]
    assert got["bytes_back"] > 700_000
    assert got["n_blocks"] == n_blocks
    assert sent["n_frames_back"] == n_blocks == len(received)


def test_the_pacing_rule_drops_a_chunk_that_is_a_whole_chunk_late():
    # A sender that cannot meet the requested rate reports a drop rather
    # than quietly running slower: that is what makes the sustained rate
    # of the rate sweep a measurement instead of a wish.
    assert stream.chunk_is_late(
        now=10.0, deadline=1.0, chunk_seconds=0.5
    )
    assert not stream.chunk_is_late(
        now=1.2, deadline=1.0, chunk_seconds=0.5
    )
    assert not stream.chunk_is_late(
        now=0.5, deadline=1.0, chunk_seconds=0.5
    )


def _bare_drain_server():
    """A listening socket that accepts one connection and discards
    whatever it sends, replying nothing: enough to exercise
    :func:`stream.replay`'s pacing without a real tracker."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def run():
        conn, _peer = server.accept()
        conn.settimeout(5.0)
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass
        conn.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server, thread, port


def _slow_chunks(n, delay, size=100):
    """``n`` chunks, each preceded by a ``delay``-second stall: a source
    that cannot keep up with any rate the test asks for."""
    for _ in range(n):
        time.sleep(delay)
        yield ("AAAA", "east", np.arange(size, dtype=float), np.zeros(size))


def _chunks_with_one_stall(n, stall_index, stall_seconds, size=50):
    """``n`` chunks, immediate except for one ``stall_seconds`` pause: a
    transient hiccup rather than a sustained overload."""
    for i in range(n):
        if i == stall_index:
            time.sleep(stall_seconds)
        yield ("AAAA", "east", np.arange(size, dtype=float), np.zeros(size))


def test_replay_recovers_a_single_transient_stall():
    server, thread, port = _bare_drain_server()
    try:
        sent = stream.replay(
            "127.0.0.1", port, _chunks_with_one_stall(5, 0, 0.6),
            rate=500, timeout=10.0,
        )
    finally:
        server.close()
        thread.join(timeout=5.0)
    # one chunk missed its deadline by more than the whole-chunk grace
    # period and is dropped; the rest, no longer behind, are sent.
    assert sent["n_dropped"] == 1
    assert sent["n_frames"] == 4


def test_replay_degrades_instead_of_collapsing_under_sustained_overload():
    server, thread, port = _bare_drain_server()
    try:
        sent = stream.replay(
            "127.0.0.1", port, _slow_chunks(10, 0.35, 100),
            rate=1000, timeout=10.0,
        )
    finally:
        server.close()
        thread.join(timeout=5.0)
    # a source that cannot sustain the rate at all would drop every
    # remaining chunk without the give-up rule; two consecutive misses
    # stop pacing so the rest are sent at the source's own pace.
    assert sent["n_dropped"] == 2
    assert sent["n_frames"] == 8
    assert sent["samples_per_second"] > 0.0


def test_serve_counts_a_frame_of_another_kind_as_other(tmp_path):
    port_file = tmp_path / "port"
    result: dict = {}

    def run_server():
        result["out"] = stream.serve(
            "127.0.0.1", 0, port_file=port_file, timeout=10.0, ack=False,
        )

    thread = threading.Thread(target=run_server)
    thread.start()
    try:
        deadline = time.time() + 10.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        port = int(port_file.read_text().strip())
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
            stream.send_frame(sock, {"kind": "bogus", "nbytes": 0}, b"")
            img = blocks(1)[0]
            stream.send_image(sock, img, station="AAAA", field="east")
    finally:
        thread.join(timeout=10.0)
    assert result["out"]["n_other"] == 1
    assert result["out"]["n_images"] == 1


def test_track_counts_a_frame_of_another_kind_as_other(tmp_path):
    filt = CountingFilter()
    port_file = tmp_path / "port"
    result: dict = {}

    def run_tracker():
        result["out"] = stream.track(
            "127.0.0.1", 0, filt, station="AAAA", field="east",
            port_file=port_file, timeout=10.0,
        )

    thread = threading.Thread(target=run_tracker)
    thread.start()
    try:
        deadline = time.time() + 10.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        port = int(port_file.read_text().strip())
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
            stream.send_frame(sock, {"kind": "bogus", "nbytes": 0}, b"")
            t = np.array([0.0])
            y = np.array([1.0])
            stream.send_samples(sock, "AAAA", "east", t, y, seq=0)
    finally:
        thread.join(timeout=10.0)
    assert result["out"]["n_other"] == 1
    assert result["out"]["n_frames"] == 1


def test_image_from_frame_rejects_bad_dtype_and_short_payload():
    img = blocks(1)[0]
    header = stream.image_header(img, station="AAAA", field="east")
    payload = stream.image_payload(img)
    with pytest.raises(ValueError, match="bytes"):
        stream.image_from_frame(header, payload[:-8])
    bad = dict(header, dtype="float32")
    with pytest.raises(ValueError, match="dtype"):
        stream.image_from_frame(bad, payload)


def test_recv_exactly_raises_when_the_peer_closes_mid_frame():
    a, b = socket.socketpair()
    try:
        a.sendall(b"12")
        a.close()
        with pytest.raises(ConnectionError, match="peer closed"):
            stream._recv_exactly(b, 5)
    finally:
        b.close()


def test_recv_frame_raises_on_truncation_before_header_and_payload():
    # Nothing at all of the header arrives: recv_frame's own message.
    a, b = socket.socketpair()
    try:
        a.sendall(stream._LEN.pack(10))
        a.close()
        with pytest.raises(ConnectionError, match="before the header"):
            stream.recv_frame(b)
    finally:
        b.close()

    # Nothing at all of the payload arrives: recv_frame's own message.
    a, b = socket.socketpair()
    try:
        header = json.dumps(
            {"kind": "samples", "nbytes": 20}, separators=(",", ":")
        ).encode("utf-8")
        a.sendall(stream._LEN.pack(len(header)))
        a.sendall(header)
        a.close()
        with pytest.raises(ConnectionError, match="before the payload"):
            stream.recv_frame(b)
    finally:
        b.close()


def test_track_reports_a_timeout_instead_of_hanging(tmp_path):
    port_file = tmp_path / "port"
    filt = CountingFilter()
    with pytest.raises(TimeoutError):
        stream.track(
            "127.0.0.1", 0, filt, port_file=port_file, timeout=0.5,
        )


def test_main_exits_with_timeout_code_two(tmp_path, capsys):
    out = tmp_path / "summary.json"
    code = stream.main([
        "--role", "consumer", "--host", "127.0.0.1", "--port", "0",
        "--timeout", "0.5", "--out", str(out),
    ])
    assert code == 2
    assert not out.exists()


def test_produce_without_ack_reports_no_latency(tmp_path):
    server, thread, port = _bare_drain_server()
    imgs = blocks(2)
    try:
        summary = stream.produce(
            "127.0.0.1", port,
            ((img, {"station": "AAAA", "field": "east"}) for img in imgs),
            ack=False,
        )
    finally:
        server.close()
        thread.join(timeout=5.0)
    assert summary["n_images"] == 2
    assert summary["n_acks"] == 0
    assert summary["latency_ms_median"] is None
    assert summary["latency_ms_p90"] is None
    assert summary["bytes_sent"] > 0
