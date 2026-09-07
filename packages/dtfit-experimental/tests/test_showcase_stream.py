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
        sent = stream.replay("127.0.0.1", port, chunks, rate=None)
    finally:
        thread.join(timeout=30.0)
    got = result["out"]
    assert sent["n_frames"] == 15 and sent["n_dropped"] == 0
    assert sent["n_samples"] == 300
    assert got["n_samples"] == 300 and got["n_dropped"] == 0
    assert got["n_flags"] == 6                # every 50th of 300 samples
    assert got["us_per_update"] > 0.0
    assert got["n_blocks"] == 3 and got["bytes_back"] > 0
    assert sent["n_frames_back"] == got["n_blocks"]
    assert got["params"]["c"] == pytest.approx(float(y[-1]))


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
