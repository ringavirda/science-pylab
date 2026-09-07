"""Leg 5: the image and the samples over plain TCP, in both directions.

A frame is a 4-byte big-endian header length, a UTF-8 JSON header, then a
payload whose shape the header's ``kind`` fixes:

- ``"image"``: ``S``, then ``G`` row-major, then the weights when the
  image carries them, all float64. The header describes the image
  completely (basis, order, domain, counts, sums and the grid), so the
  receiver rebuilds an :class:`~dtfit.image.Image` that compares equal to
  the sender's.
- ``"samples"``: ``t`` then ``y``, float64, ``n`` values each.
- ``"ack"``: no payload. The consumer sends one per image frame and the
  producer records the round trip, which is a per-block latency that
  needs no clock synchronisation between the machines.

Every receiver dispatches on ``kind``, so a frame of another kind on the
same socket is counted rather than misread.

Nothing here knows about the datasets or the filters: the consumer's model
arrives as an expression string and a name list, the tracker's filter as
an object with ``partial_fit``, ``drift_flag_`` and ``params_``. The
dataset wiring is in ``cli.py`` (``stream-serve``, ``stream-replay``,
``stream-track``).

Running it: the consumer binds and waits, the producer walks a directory
of reduced ``.npz`` files and sends their block images.

    python -m dtfit_experimental.experiments.domains.image_showcase.stream \\
        --role consumer --host 0.0.0.0 --port 5555 --out summary.json
    python -m dtfit_experimental.experiments.domains.image_showcase.stream \\
        --role producer --host felled-pi.local --port 5555 --images DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from dtfit.image import Grid, Image, assemble, make_basis

from .compare import fit_from_image
from .store import load_images

MODULE = "dtfit_experimental.experiments.domains.image_showcase.stream"
IMAGE_KIND = "image"
SAMPLE_KIND = "samples"
ACK_KIND = "ack"
_LEN = struct.Struct(">I")
# A block's domain start and a stored drift flag's domain start are the
# same float; this is the slack a join on them allows.
_DOMAIN_TOL = 1e-9


def _recv_exactly(sock: socket.socket, count: int) -> bytes | None:
    """``count`` bytes from ``sock``; ``None`` when the peer closed before
    sending any of them.

    Raises:
        ConnectionError: the peer closed part way through.
    """
    chunks: list[bytes] = []
    got = 0
    while got < count:
        block = sock.recv(count - got)
        if not block:
            if got == 0:
                return None
            raise ConnectionError(
                f"peer closed after {got} of {count} bytes"
            )
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket, header: dict[str, Any], payload: bytes
) -> int:
    """Send one frame; returns the bytes written (length prefix, header
    and payload)."""
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(_LEN.pack(len(blob)))
    sock.sendall(blob)
    if payload:
        sock.sendall(payload)
    return _LEN.size + len(blob) + len(payload)


def recv_frame(
    sock: socket.socket,
) -> tuple[dict[str, Any], bytes] | None:
    """Receive one frame as ``(header, payload)``, or ``None`` at a clean
    end of stream. The header's ``nbytes`` gives the payload length.

    Raises:
        ConnectionError: the peer closed mid-frame.
    """
    head = _recv_exactly(sock, _LEN.size)
    if head is None:
        return None
    (size,) = _LEN.unpack(head)
    blob = _recv_exactly(sock, size)
    if blob is None:
        raise ConnectionError("peer closed before the header")
    header = json.loads(blob.decode("utf-8"))
    nbytes = int(header.get("nbytes", 0))
    payload = b""
    if nbytes:
        got = _recv_exactly(sock, nbytes)
        if got is None:
            raise ConnectionError("peer closed before the payload")
        payload = got
    return header, payload


def image_payload(image: Image) -> bytes:
    """``S``, ``G`` and any weights as C-order float64 bytes."""
    parts = [
        np.ascontiguousarray(image.S, dtype=np.float64).tobytes(),
        np.ascontiguousarray(image.G, dtype=np.float64).tobytes(),
    ]
    if image.w is not None:
        parts.append(
            np.ascontiguousarray(image.w, dtype=np.float64).tobytes()
        )
    return b"".join(parts)


def image_header(image: Image, **extra: Any) -> dict[str, Any]:
    """The frame header of one image.

    Carries everything but the arrays: ``kind``, the basis name and order,
    the domain, ``n``, ``sumsq``, ``sumy``, ``wsum``, ``robust``, the grid
    (``Grid.to_dict()``, which for an explicit grid holds every position),
    the payload dtype, the array shapes, ``nbytes`` and ``sent_at`` (the
    sender's wall clock). ``extra`` adds the station, field and block
    index the consumer groups by.
    """
    k = int(image.S.size)
    shapes: dict[str, Any] = {"S": [k], "G": [k, k], "w": None}
    if image.w is not None:
        shapes["w"] = [int(image.w.size)]
    header: dict[str, Any] = {
        "kind": IMAGE_KIND,
        "basis": image.basis.name,
        "order": int(image.order),
        "domain": [float(image.domain[0]), float(image.domain[1])],
        "n": int(image.n),
        "sumsq": float(image.sumsq),
        "sumy": float(image.sumy),
        "wsum": float(image.wsum),
        "robust": bool(image.robust),
        "grid": image.grid.to_dict(),
        "dtype": "float64",
        "shapes": shapes,
        "sent_at": time.time(),
    }
    header.update(extra)
    header["nbytes"] = (k + k * k) * 8 + (
        0 if image.w is None else int(image.w.size) * 8
    )
    return header


def image_from_frame(header: dict[str, Any], payload: bytes) -> Image:
    """Rebuild the image a frame carries.

    Raises:
        ValueError: the payload is shorter than the header's shapes, or
            the dtype is not float64.
    """
    if header.get("dtype") != "float64":
        raise ValueError(f"unsupported dtype {header.get('dtype')!r}")
    k = int(header["shapes"]["S"][0])
    need = (k + k * k) * 8
    w_shape = header["shapes"].get("w")
    if w_shape is not None:
        need += int(w_shape[0]) * 8
    if len(payload) < need:
        raise ValueError(
            f"payload has {len(payload)} bytes, the header needs {need}"
        )
    S = np.frombuffer(payload, dtype=np.float64, count=k).copy()
    G = np.frombuffer(
        payload, dtype=np.float64, count=k * k, offset=k * 8
    ).reshape(k, k).copy()
    w = None
    if w_shape is not None:
        w = np.frombuffer(
            payload, dtype=np.float64, count=int(w_shape[0]),
            offset=(k + k * k) * 8,
        ).copy()
    grid = Grid.from_dict(dict(header["grid"]))
    return Image(
        make_basis(header["basis"], int(header["order"])),
        (float(header["domain"][0]), float(header["domain"][1])),
        S, G, int(header["n"]), float(header["sumsq"]),
        float(header["sumy"]), float(header["wsum"]), grid, w,
        bool(header["robust"]),
    )


def send_image(sock: socket.socket, image: Image, **extra: Any) -> int:
    """Send one image as a frame; returns the bytes written."""
    return send_frame(sock, image_header(image, **extra),
                      image_payload(image))


def send_ack(sock: socket.socket, seq: Any) -> int:
    """Acknowledge the frame numbered ``seq``; returns the bytes
    written. The sender's round trip on this is the per-block latency,
    which needs no clock synchronisation between the machines."""
    return send_frame(sock, {"kind": ACK_KIND, "seq": seq,
                             "nbytes": 0}, b"")


def samples_payload(t: Any, y: Any) -> bytes:
    """``t`` then ``y`` as C-order float64 bytes, ``2 * n * 8`` long."""
    return (
        np.ascontiguousarray(t, dtype=np.float64).tobytes()
        + np.ascontiguousarray(y, dtype=np.float64).tobytes()
    )


def samples_header(
    station: str, field: str, n: int, *, seq: int, **extra: Any
) -> dict[str, Any]:
    """The frame header of one raw-sample chunk.

    ``seq`` counts chunks at the sender and increments for a chunk the
    sender skipped as well, so the receiver sees the gap and can count
    the drops independently.
    """
    header: dict[str, Any] = {
        "kind": SAMPLE_KIND,
        "station": str(station),
        "field": str(field),
        "seq": int(seq),
        "n": int(n),
        "dtype": "float64",
        "shapes": {"t": [int(n)], "y": [int(n)]},
        "sent_at": time.time(),
    }
    header.update(extra)
    header["nbytes"] = 2 * int(n) * 8
    return header


def samples_from_frame(
    header: dict[str, Any], payload: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` from a ``samples`` frame.

    Raises:
        ValueError: the payload is shorter than ``2 * n * 8``, or the
            dtype is not float64.
    """
    if header.get("dtype") != "float64":
        raise ValueError(f"unsupported dtype {header.get('dtype')!r}")
    n = int(header["n"])
    need = 2 * n * 8
    if len(payload) < need:
        raise ValueError(
            f"payload has {len(payload)} bytes, the header needs {need}"
        )
    t = np.frombuffer(payload, dtype=np.float64, count=n).copy()
    y = np.frombuffer(
        payload, dtype=np.float64, count=n, offset=n * 8
    ).copy()
    return t, y


def send_samples(
    sock: socket.socket,
    station: str,
    field: str,
    t: Any,
    y: Any,
    *,
    seq: int,
    **extra: Any,
) -> int:
    """Send one raw-sample chunk as a frame; returns the bytes written."""
    n = int(np.asarray(t).size)
    return send_frame(
        sock, samples_header(station, field, n, seq=seq, **extra),
        samples_payload(t, y),
    )


def chunk_is_late(
    now: float, deadline: float, chunk_seconds: float
) -> bool:
    """Whether a paced sender is more than one whole chunk behind.

    ``now`` and ``deadline`` are ``time.perf_counter`` readings and
    ``chunk_seconds`` is the chunk's nominal duration at the requested
    rate. A sender that is this far behind skips the chunk and counts a
    drop rather than sending late, which is what turns "the rate it
    sustained" into a measurement.
    """
    return float(now) - float(deadline) > float(chunk_seconds)


def image_digest(image: Image) -> str:
    """SHA-256 over ``S`` and ``G`` as float64 bytes: two images agree bit
    for bit exactly when their digests match."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(image.S, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(image.G, dtype=np.float64).tobytes())
    return h.hexdigest()


def _flag_starts(info: dict[str, Any], field: str) -> list[float]:
    """The domain starts of the drift flags a reducer stored for one
    field. NGL keeps them per component in ``info["flags"]``, NOAA in a
    flat ``info["day_flags"]``; both are ``[index, t0, t1]`` triples."""
    flags = info.get("flags")
    entries: list[Any] = []
    if isinstance(flags, dict):
        entries = list(flags.get(field, []))
    elif isinstance(flags, list):
        entries = list(flags)
    entries += list(info.get("day_flags", []))
    return [float(e[1]) for e in entries if len(e) >= 2]


def iter_images(
    images_dir: Any, *, prefix: str = "blk", limit: int | None = None
) -> Iterator[tuple[Image, dict[str, Any]]]:
    """Walk ``<images_dir>/*.npz`` and yield the images whose stored name
    starts with ``prefix``, each with
    ``{"station", "field", "block", "flags"}``.

    The name is ``<prefix><index>_<field>``; a name without an index gives
    ``block = -1``. ``flags`` is how many drift flags the reducer's stream
    raised on that block, joined on the block's domain start rather than
    on any index, because a stream's internal index also advances for the
    blocks it drops. Files are visited in sorted order and images in
    stored order, so a run is reproducible. ``limit`` stops after that
    many images.
    """
    sent = 0
    for path in sorted(Path(images_dir).glob("*.npz")):
        images, info = load_images(path)
        station = str(info.get("sta", info.get("station", path.stem)))
        for name, image in images.items():
            if not name.startswith(prefix):
                continue
            head, _, field = name.partition("_")
            digits = head[len(prefix):]
            starts = _flag_starts(info, field)
            t0 = float(image.domain[0])
            yield image, {
                "station": station, "field": field,
                "block": int(digits) if digits.isdigit() else -1,
                "flags": sum(
                    1 for s in starts if abs(s - t0) <= _DOMAIN_TOL
                ),
            }
            sent += 1
            if limit is not None and sent >= limit:
                return


def produce(
    host: str,
    port: int,
    items: Iterable[tuple[Image, dict[str, Any]]],
    *,
    timeout: float = 30.0,
    ack: bool = True,
) -> dict[str, Any]:
    """Connect and send every ``(image, extra)`` as an image frame.

    With ``ack`` the producer waits for the consumer's ``ack`` frame
    before sending the next image and records the round trip, so
    ``latency_ms_median`` and ``latency_ms_p90`` are real per-block
    latencies rather than a clock difference. That also serialises the
    stream, so ``images_per_second`` is the acked rate; pass
    ``ack=False`` for an unacked upper bound.

    Returns:
        ``{"n_images", "bytes_sent", "seconds", "images_per_second",
        "n_acks", "latency_ms_median", "latency_ms_p90"}``; the two
        latencies are ``None`` without acks.
    """
    started = time.perf_counter()
    total = 0
    count = 0
    acks = 0
    latencies: list[float] = []
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        for image, extra in items:
            at = time.perf_counter()
            total += send_image(sock, image, seq=count, **extra)
            count += 1
            if ack:
                frame = recv_frame(sock)
                if frame is None:
                    break
                if frame[0].get("kind") == ACK_KIND:
                    acks += 1
                    latencies.append((time.perf_counter() - at) * 1000.0)
    seconds = time.perf_counter() - started
    return {
        "n_images": count, "bytes_sent": total,
        "seconds": round(seconds, 4),
        "images_per_second": round(count / seconds, 3) if seconds else 0.0,
        "n_acks": acks,
        "latency_ms_median": (
            round(float(np.median(latencies)), 4) if latencies else None
        ),
        "latency_ms_p90": (
            round(float(np.percentile(latencies, 90)), 4)
            if latencies else None
        ),
    }


def _listen(
    host: str, port: int, port_file: Any, timeout: float
) -> tuple[socket.socket, socket.socket]:
    """Bind, publish the bound port and accept one connection.

    Returns ``(listener, connection)``; the caller closes both.
    ``port_file`` receives the bound port, which lets a caller pass
    ``port=0`` and discover it.

    Raises:
        TimeoutError: no connection within ``timeout`` seconds.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(1)
    listener.settimeout(timeout)
    bound = listener.getsockname()[1]
    if port_file is not None:
        # Written via a temp path and renamed so a poller that only checks
        # existence never observes a half-written file.
        port_file = Path(port_file)
        tmp = port_file.with_name(port_file.name + ".tmp")
        tmp.write_text(f"{bound}\n")
        tmp.replace(port_file)
    try:
        conn, _peer = listener.accept()
    except socket.timeout as exc:
        listener.close()
        raise TimeoutError(
            f"no connection within {timeout} seconds"
        ) from exc
    return listener, conn


def serve(
    host: str,
    port: int,
    *,
    expect: int | None = None,
    port_file: Any = None,
    timeout: float = 60.0,
    expr: str | None = None,
    names: Sequence[str] | None = None,
    var: str = "t",
    ack: bool = True,
) -> dict[str, Any]:
    """Accept one connection and consume frames until the peer closes or
    ``expect`` image frames have arrived.

    Image frames are grouped by ``"<station>/<field>"`` and each group is
    assembled with :func:`dtfit.image.assemble` at the blocks' own order.
    Per group the summary carries the assembly's digest (so the far
    machine's assembly can be compared bit for bit against the near
    one's), the header bytes, payload bytes and sample count, the number
    of drift flags the producer's stream raised on those blocks, and,
    when ``expr`` and ``names`` are given, the parameters of a fit of that
    model to the assembled image: the analysis itself, run on the
    receiving machine. With ``ack`` each image frame is acknowledged, so
    the producer can time the round trip. A frame of any other kind is
    counted in ``n_other`` and skipped.

    Raises:
        TimeoutError: no connection or no frame within ``timeout``
            seconds.
    """
    groups: dict[str, list[Image]] = {}
    group_bytes: dict[str, list[int]] = {}
    group_flags: dict[str, int] = {}
    bytes_header = 0
    bytes_payload = 0
    deltas: list[float] = []
    count = 0
    other = 0
    listener, conn = _listen(host, port, port_file, timeout)
    started = time.perf_counter()
    try:
        with conn:
            conn.settimeout(timeout)
            while expect is None or count < expect:
                try:
                    frame = recv_frame(conn)
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"no frame within {timeout} seconds"
                    ) from exc
                if frame is None:
                    break
                header, payload = frame
                now = time.time()
                head_bytes = _LEN.size + len(
                    json.dumps(header, separators=(",", ":"))
                    .encode("utf-8")
                )
                bytes_header += head_bytes
                bytes_payload += len(payload)
                if header.get("kind") != IMAGE_KIND:
                    other += 1
                    continue
                deltas.append(now - float(header.get("sent_at", now)))
                key = f"{header.get('station', '?')}/" \
                      f"{header.get('field', '?')}"
                image = image_from_frame(header, payload)
                groups.setdefault(key, []).append(image)
                tally = group_bytes.setdefault(key, [0, 0, 0])
                tally[0] += head_bytes
                tally[1] += len(payload)
                tally[2] += int(image.n)
                group_flags[key] = group_flags.get(key, 0) + int(
                    header.get("flags", 0)
                )
                count += 1
                if ack:
                    send_ack(conn, header.get("seq"))
    finally:
        listener.close()
    seconds = time.perf_counter() - started
    summary: dict[str, Any] = {
        "n_images": count, "n_other": other,
        "bytes_header": bytes_header,
        "bytes_payload": bytes_payload,
        "seconds": round(seconds, 4),
        "images_per_second": (
            round(count / seconds, 3) if seconds else 0.0
        ),
        # Valid only between clocks synchronised by NTP; the run reports
        # it as a diagnostic. The latency claim is the producer's acked
        # round trip.
        "clock_delta_s": (
            round(float(np.median(deltas)), 4) if deltas else None
        ),
        "groups": {},
    }
    for key, images in groups.items():
        whole = assemble(images, order=min(i.order for i in images))
        head, payload_bytes, samples = group_bytes[key]
        entry: dict[str, Any] = {
            "n_blocks": len(images), "order": int(whole.order),
            "domain": [float(whole.domain[0]), float(whole.domain[1])],
            "n": int(whole.n), "digest": image_digest(whole),
            "bytes_header": head, "bytes_payload": payload_bytes,
            "samples": samples, "n_flags": group_flags.get(key, 0),
            "params": None, "rss": None, "converged": None,
        }
        if expr is not None and names:
            res = fit_from_image(expr, whole, list(names), var=var)
            entry["params"] = {
                k: float(v) for k, v in res.params.items()
            }
            entry["rss"] = (
                None if res.rss is None else float(res.rss)
            )
            entry["converged"] = bool(res.converged)
        summary["groups"][key] = entry
    return summary


def replay(
    host: str,
    port: int,
    chunks: Iterable[tuple[str, str, Any, Any]],
    *,
    rate: float | None = None,
    timeout: float = 30.0,
    back_sink: Callable[[dict[str, Any], bytes], None] | None = None,
) -> dict[str, Any]:
    """Send ``(station, field, t, y)`` chunks as ``samples`` frames.

    ``rate`` is the nominal sample rate in samples per second (``None``
    sends as fast as the link allows). Pacing keeps a deadline that
    advances by ``n / rate`` per chunk; when the replayer falls more than
    one chunk behind (:func:`chunk_is_late`), the chunk is skipped rather
    than sent and ``n_dropped`` counts it. Two such misses in a row mean
    the source cannot sustain the requested rate at all -- pacing then
    stops for the rest of the run and every later chunk is sent at
    whatever pace the source allows, so the achieved rate is what gets
    measured instead of every remaining chunk dropping to zero. ``seq``
    increments on every chunk including the skipped ones, so the receiver
    sees the gap and counts the drops independently.

    A background thread drains the tracker's return traffic concurrently
    with sending, so a busy back channel (block images) can never fill
    this socket's receive buffer and deadlock both ends in ``sendall``.
    ``back_sink``, when given, is called with every returned
    ``(header, payload)`` frame; otherwise the frames are only counted.
    After the last chunk the write side is shut down; the socket is
    still read to end of stream, so the tracker's final block images
    arrive and are counted in ``n_frames_back``.

    Returns:
        ``{"n_frames", "n_samples", "n_dropped", "bytes_sent", "seconds",
        "samples_per_second", "n_frames_back", "rate_requested"}``.

    Raises:
        ConnectionError: the return channel closed mid-frame.
    """
    n_frames = 0
    n_samples = 0
    n_dropped = 0
    n_back = 0
    total = 0
    seq = 0
    back_errors: list[BaseException] = []
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)

        def drain() -> None:
            nonlocal n_back
            try:
                while True:
                    frame = recv_frame(sock)
                    if frame is None:
                        return
                    n_back += 1
                    if back_sink is not None:
                        back_sink(*frame)
            except OSError as exc:
                back_errors.append(exc)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        deadline = time.perf_counter()
        consecutive_drops = 0
        paced = True
        for station, field, t, y in chunks:
            n = int(np.asarray(t).size)
            if paced and rate:
                span = n / float(rate)
                deadline += span
                now = time.perf_counter()
                if chunk_is_late(now, deadline, span):
                    n_dropped += 1
                    seq += 1
                    consecutive_drops += 1
                    deadline = now
                    if consecutive_drops >= 2:
                        paced = False
                    continue
                consecutive_drops = 0
                if now < deadline:
                    time.sleep(deadline - now)
            total += send_samples(sock, station, field, t, y, seq=seq)
            seq += 1
            n_frames += 1
            n_samples += n
        sock.shutdown(socket.SHUT_WR)
        reader.join(timeout=timeout)
    if back_errors:
        raise back_errors[0]
    seconds = time.perf_counter() - started
    return {
        "n_frames": n_frames, "n_samples": n_samples,
        "n_dropped": n_dropped, "bytes_sent": total,
        "seconds": round(seconds, 4),
        "samples_per_second": (
            round(n_samples / seconds, 1) if seconds else 0.0
        ),
        "n_frames_back": n_back,
        "rate_requested": None if rate is None else float(rate),
    }


def track(
    host: str,
    port: int,
    filt: Any,
    *,
    block_stream: Any = None,
    back: bool = True,
    station: str = "",
    field: str = "",
    port_file: Any = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Accept one connection and feed every ``samples`` frame into a
    filter, sending block images back over the same socket.

    ``filt`` is any object with ``partial_fit(t, y)``, ``drift_flag_``
    and ``params_`` -- an :class:`~dtfit.streaming.LSIFilter` in the run,
    built by the caller so that this module needs no filter import.
    ``block_stream`` is an :class:`~dtfit.image.ImageStream` in block
    mode; each block it finishes is sent back as an ``image`` frame when
    ``back``, including the blocks its ``close()`` yields after the peer
    has shut down its write side.

    ``n_dropped`` counts a gap between ``seq`` values this end actually
    received, an interior-drop check against the sender's own count. A
    run of drops right at the end of the stream, with nothing arriving
    after them, is invisible here -- use the sender's own ``n_dropped``
    for the true total. ``us_per_update`` times the filter loop alone,
    not the socket.

    Returns:
        ``{"n_frames", "n_samples", "n_dropped", "n_other", "seconds",
        "samples_per_second", "us_per_update", "n_flags", "flags",
        "n_blocks", "bytes_back", "params"}``.

    Raises:
        TimeoutError: no connection or no frame within ``timeout``
            seconds.
    """
    listener, conn = _listen(host, port, port_file, timeout)
    n_frames = 0
    n_samples = 0
    n_dropped = 0
    n_other = 0
    n_blocks = 0
    bytes_back = 0
    update_seconds = 0.0
    flags: list[float] = []
    expected = 0
    started = time.perf_counter()

    def ship(images: list[Any]) -> int:
        sent = 0
        for img in images:
            sent += send_image(
                conn, img, station=station, field=field,
                block=int(round(float(img.domain[0]))), seq=n_blocks,
                flags=0,
            )
        return sent

    try:
        with conn:
            conn.settimeout(timeout)
            while True:
                try:
                    frame = recv_frame(conn)
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"no frame within {timeout} seconds"
                    ) from exc
                if frame is None:
                    break
                header, payload = frame
                if header.get("kind") != SAMPLE_KIND:
                    n_other += 1
                    continue
                seq = int(header.get("seq", expected))
                if seq > expected:
                    n_dropped += seq - expected
                expected = seq + 1
                t, y = samples_from_frame(header, payload)
                n_frames += 1
                n_samples += int(t.size)
                at = time.perf_counter()
                for k in range(t.size):
                    filt.partial_fit(float(t[k]), float(y[k]))
                    if filt.drift_flag_:
                        flags.append(float(t[k]))
                update_seconds += time.perf_counter() - at
                if block_stream is not None:
                    finished = block_stream.update(t, y)
                    if finished:
                        n_blocks += len(finished)
                        if back:
                            bytes_back += ship(finished)
            if block_stream is not None:
                last = block_stream.close()
                if last:
                    n_blocks += len(last)
                    if back:
                        bytes_back += ship(last)
    finally:
        listener.close()
    seconds = time.perf_counter() - started
    return {
        "n_frames": n_frames, "n_samples": n_samples,
        "n_dropped": n_dropped, "n_other": n_other,
        "seconds": round(seconds, 4),
        "samples_per_second": (
            round(n_samples / seconds, 1) if seconds else 0.0
        ),
        "us_per_update": (
            round(update_seconds / n_samples * 1e6, 3)
            if n_samples else 0.0
        ),
        "n_flags": len(flags), "flags": flags,
        "n_blocks": n_blocks, "bytes_back": bytes_back,
        "params": {k: float(v) for k, v in dict(filt.params_).items()},
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the two image roles; returns a
    process exit code (0 on success, 2 on a timeout).

    The consumer takes its model as ``--expr`` and ``--names`` rather
    than naming a dataset, so this module stays free of the dataset
    modules; ``cli.py``'s ``stream-serve``, ``stream-replay`` and
    ``stream-track`` fill those in and build the tracker's filter.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("producer", "consumer"),
                        required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--images", default=None,
                        help="producer: directory of reduced .npz files")
    parser.add_argument("--prefix", default="blk")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expect", type=int, default=None)
    parser.add_argument("--port-file", default=None)
    parser.add_argument("--out", default=None,
                        help="write the summary as JSON here")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--expr", default=None,
                        help="consumer: the model to fit on each "
                             "assembled image")
    parser.add_argument("--names", default=None,
                        help="consumer: its parameter names, comma "
                             "separated, in canonical order")
    parser.add_argument("--var", default="t")
    parser.add_argument("--no-ack", action="store_true",
                        help="do not acknowledge each image frame")
    args = parser.parse_args(argv)
    try:
        if args.role == "producer":
            if args.images is None:
                parser.error("--images is required for the producer")
            summary = produce(
                args.host, args.port,
                iter_images(args.images, prefix=args.prefix,
                            limit=args.limit),
                timeout=args.timeout, ack=not args.no_ack,
            )
        else:
            summary = serve(
                args.host, args.port, expect=args.expect,
                port_file=args.port_file, timeout=args.timeout,
                expr=args.expr,
                names=args.names.split(",") if args.names else None,
                var=args.var, ack=not args.no_ack,
            )
    except TimeoutError as exc:
        print(f"timeout: {exc}")
        return 2
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
