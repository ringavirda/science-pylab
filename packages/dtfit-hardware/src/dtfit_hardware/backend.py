"""Host-side control and telemetry for the realtime_gps_hw rig.

Talks to the Arduino Nano 33 BLE Sense. Over USB it locates the board, flashes
a firmware sketch from ``firmware/`` and captures the serial telemetry; over
BLE it receives the same stream untethered, which is what lets the rig run on
battery in open sky.

The toolchain is ``arduino-cli``, taken from PATH or from an Arduino IDE
install, so a machine with the IDE needs no extra download. ``pyserial`` and
``bleak`` carry the two links and are ordinary dependencies of this package.

CLI (board on USB / advertising over BLE)::

    python backend.py find                  # detect board + serial port
    python backend.py selftest              # flash diagnostic, verify IMU
    python backend.py flash nano_gps_parse
    python backend.py capture 10            # print 10 s of USB serial
    python backend.py log data/run1.csv 60  # log 60 s of USB serial to CSV
    python backend.py ble 20                # receive 20 s of BLE telemetry
    python backend.py blelog data/r.csv 120 # log 120 s of BLE telemetry to CSV
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

FQBN = "arduino:mbed_nano:nano33ble"
DEFAULT_BAUD = 115200

# BLE identifiers; must match the UUIDs in the firmware sketches.
BLE_NAME = "dtfit-gps"
BLE_SERVICE_UUID = "9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10"
BLE_TELE_UUID = "9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10"
BLE_CSV_HEADER = (
    "t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz"
)
# nano_lsi_log emits a wider record: the raw fix and IMU, then the on-MCU
# float32 LSI estimate and 1 s forecast (``est_*``/``fc_*``, computed in
# local-ENU metres to keep float32 well-conditioned), the IMU-adaptive ``mode``
# (0 still, 1 moving), the per-update cost in microseconds, the heading block,
# ``imu_hz`` (IMU samples integrated this interval, which confirms the rate),
# ``newfix`` (1 on a fresh GPS fix, 0 on a between-fix sample) and ``sd``, the
# write health the untethered phone renders as REC / NO SD.
#
# Of the heading pair, ``hdg_deg`` is anchored to the GPS course on-chip: it is
# drift-free but not independent of GPS. ``dhdg_deg`` is the raw
# gravity-aligned yaw increment since the last emit, and that is the one the
# host dead-reckons on. The host scores the float32 ``est_*`` columns against a
# float64 LSI replayed on the raw lat/lon.
BLE_CSV_HEADER_LSI = (
    "t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz,mx,my,mz,"
    "est_lat,est_lon,fc_lat,fc_lon,mode,cost_us,hdg_deg,dhdg_deg,imu_hz,newfix,sd"
)

HERE = Path(__file__).resolve().parent
FIRMWARE_DIR = HERE / "firmware"
DATA_DIR = HERE / "data"

# Arduino libraries the firmware depends on (arduino-cli lib install).
REQUIRED_LIBS = ("Arduino_BMI270_BMM150", "TinyGPSPlus", "ArduinoBLE", "SdFat")

# The Arduino IDE ships arduino-cli here on Windows; used if not on PATH.
_BUNDLED_CLI = (
    Path.home() / "AppData" / "Local" / "Programs" / "arduino-ide"
    / "resources" / "app" / "lib" / "backend" / "resources"
    / "arduino-cli.exe"
)


@dataclass
class Board:
    """A detected board: serial ``port``, ``fqbn`` and human ``name``."""

    port: str
    fqbn: str
    name: str


# Arduino toolchain (arduino-cli).
def arduino_cli() -> str:
    """Locate ``arduino-cli``: PATH first, then the bundled Arduino IDE copy."""
    exe = shutil.which("arduino-cli")
    if exe:
        return exe
    if _BUNDLED_CLI.exists():
        return str(_BUNDLED_CLI)
    raise FileNotFoundError(
        "arduino-cli not found; install the Arduino IDE or arduino-cli"
    )


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``arduino-cli <args>`` capturing text output."""
    return subprocess.run(
        [arduino_cli(), *args], capture_output=True, text=True
    )


def find_board(fqbn: str = FQBN) -> Board | None:
    """Return the first connected board matching ``fqbn`` (or ``None``)."""
    cp = _run(["board", "list", "--format", "json"])
    if cp.returncode != 0:
        return None
    data = json.loads(cp.stdout or "{}")
    for entry in data.get("detected_ports", []):
        addr = entry.get("port", {}).get("address")
        for b in entry.get("matching_boards", []) or []:
            if b.get("fqbn") == fqbn and addr:
                return Board(addr, fqbn, b.get("name", "?"))
    return None


def install_libs(libs: tuple[str, ...] = REQUIRED_LIBS) -> None:
    """Install the Arduino libraries the firmware depends on."""
    for lib in libs:
        cp = _run(["lib", "install", lib])
        if cp.returncode != 0:
            raise RuntimeError(f"lib install {lib} failed:\n{cp.stderr}")


def _sketch_dir(sketch: str) -> Path:
    sk = FIRMWARE_DIR / sketch
    if not (sk / f"{sketch}.ino").exists():
        raise FileNotFoundError(f"no sketch at {sk / (sketch + '.ino')}")
    return sk


def compile_sketch(sketch: str, fqbn: str = FQBN) -> None:
    """Compile ``firmware/<sketch>/`` for ``fqbn`` (no upload)."""
    cp = _run(["compile", "--fqbn", fqbn, str(_sketch_dir(sketch))])
    if cp.returncode != 0:
        raise RuntimeError(f"compile failed:\n{cp.stdout}\n{cp.stderr}")


def flash(sketch: str, port: str | None = None, fqbn: str = FQBN) -> str:
    """Compile + upload ``firmware/<sketch>/``; return the port used."""
    sk = _sketch_dir(sketch)
    compile_sketch(sketch, fqbn)
    if port is None:
        b = find_board(fqbn)
        if b is None:
            raise RuntimeError("no board found to upload to")
        port = b.port
    cp = _run(["upload", "-p", port, "--fqbn", fqbn, str(sk)])
    if cp.returncode != 0:
        raise RuntimeError(f"upload failed:\n{cp.stdout}\n{cp.stderr}")
    # After an upload the board resets and re-enumerates, sometimes on a
    # different COM port. Re-detect, or the caller gets a stale port.
    time.sleep(2.0)
    b = find_board(fqbn)
    return b.port if b else port


# USB-serial telemetry (pyserial).
def _resolve_port(port: str | None) -> str:
    if port is None:
        b = find_board()
        port = b.port if b else None
    if port is None:
        raise RuntimeError("no serial port (board not detected)")
    return port


def wait_for_port(port: str, timeout: float = 10.0) -> bool:
    """Block until ``port`` can be opened (after an upload re-enumerates)."""
    import serial  # pyserial; lazy so the module imports without it

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            serial.Serial(port).close()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _stream_lines(port: str, seconds: float, baud: int):
    """Yield ``(host_t_s, line)`` for ``seconds`` of newline-split serial."""
    import serial

    buf = b""
    with serial.Serial(port, baud, timeout=0.2) as ser:
        t0 = time.monotonic()
        deadline = t0 + seconds
        while time.monotonic() < deadline:
            chunk = ser.read(256)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                txt = raw.decode("utf-8", "replace").rstrip("\r")
                yield round(time.monotonic() - t0, 3), txt


def capture(
    port: str | None = None,
    seconds: float = 10.0,
    baud: int = DEFAULT_BAUD,
) -> list[str]:
    """Read serial lines for ``seconds`` and return them."""
    port = _resolve_port(port)
    return [line for _, line in _stream_lines(port, seconds, baud)]


def log_csv(
    path: str | Path,
    port: str | None = None,
    seconds: float = 30.0,
    baud: int = DEFAULT_BAUD,
) -> Path:
    """Capture serial for ``seconds`` to a 2-col CSV (host_t_s, line)."""
    port = _resolve_port(port)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["host_t_s", "line"])
        for t, line in _stream_lines(port, seconds, baud):
            w.writerow([t, line])
    return out


# BLE telemetry (bleak): the untethered link for battery / open-sky runs.
def _ble_stream(seconds, name, char_uuid, scan_timeout, on_line) -> None:
    """Connect to ``name`` and call ``on_line(str)`` per notification."""
    import asyncio

    from bleak import BleakClient, BleakScanner

    async def run():
        dev = await BleakScanner.find_device_by_name(
            name, timeout=scan_timeout
        )
        if dev is None:
            raise RuntimeError(
                f"BLE device {name!r} not found -- is it advertising and "
                "in range? (BLE reaches ~10 m; walls cut it down)"
            )
        async with BleakClient(dev) as client:
            def cb(_, data):
                on_line(bytes(data).decode("utf-8", "replace").strip())

            await client.start_notify(char_uuid, cb)
            await asyncio.sleep(seconds)
            try:
                await client.stop_notify(char_uuid)
            except Exception:
                pass

    asyncio.run(run())


def ble_capture(
    seconds: float = 15.0,
    name: str = BLE_NAME,
    char_uuid: str = BLE_TELE_UUID,
    scan_timeout: float = 15.0,
) -> list[str]:
    """Receive BLE telemetry lines for ``seconds`` and return them."""
    lines: list[str] = []
    _ble_stream(seconds, name, char_uuid, scan_timeout, lines.append)
    return lines


def ble_log_csv(
    path: str | Path,
    seconds: float = 60.0,
    name: str = BLE_NAME,
    char_uuid: str = BLE_TELE_UUID,
    scan_timeout: float = 15.0,
    header: str = BLE_CSV_HEADER_LSI,
) -> Path:
    """Stream BLE telemetry to a CSV, prefixed with the column header.

    The default header is ``nano_lsi_log``'s, the sketch the rig runs. Pass
    ``header=BLE_CSV_HEADER`` for the plain 14-column ``nano_ble_telemetry``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")

        def writer(line: str) -> None:
            f.write(line + "\n")
            f.flush()

        _ble_stream(seconds, name, char_uuid, scan_timeout, writer)
    return out


_NUM = r"[-+]?\d*\.?\d+"
_IMU_RE = re.compile(
    rf"a=({_NUM}),({_NUM}),({_NUM})\s+"
    rf"g=({_NUM}),({_NUM}),({_NUM})\s+"
    rf"m=({_NUM}),({_NUM}),({_NUM})"
)
_IMU_KEYS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")


def parse_imu(line: str) -> dict | None:
    """Parse a diagnostic ``IMU a=..  g=..  m=..`` line to floats, or None."""
    m = _IMU_RE.search(line)
    if not m:
        return None
    return dict(zip(_IMU_KEYS, (float(x) for x in m.groups())))


def self_test(seconds: float = 6.0, install: bool = True) -> dict:
    """Flash the diagnostic and confirm the board streams IMU telemetry.

    Returns:
        A report dict. A missing board or silent telemetry is reported as a
        soft ``ok=False``; a toolchain, compile or upload failure raises.
    """
    b = find_board()
    if b is None:
        return {"ok": False, "reason": "no board detected"}
    if install:
        install_libs(("Arduino_BMI270_BMM150",))
    flash("nano_diagnostic", port=b.port)
    wait_for_port(b.port)
    lines = capture(port=b.port, seconds=seconds)
    imu = [ln for ln in lines if parse_imu(ln)]
    return {
        "ok": len(imu) > 0,
        "port": b.port,
        "name": b.name,
        "n_lines": len(lines),
        "n_imu": len(imu),
        "sample": imu[0] if imu else (lines[-1] if lines else ""),
        "lines": lines,
    }


def _main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "selftest"
    if cmd == "find":
        b = find_board()
        print(b if b else "no board detected")
    elif cmd == "compile":
        compile_sketch(argv[1])
        print("compiled", argv[1])
    elif cmd == "flash":
        print("flashed", argv[1], "->", flash(argv[1]))
    elif cmd == "capture":
        secs = float(argv[1]) if len(argv) > 1 else 8.0
        for ln in capture(seconds=secs):
            print(ln)
    elif cmd == "log":
        path = argv[1] if len(argv) > 1 else str(DATA_DIR / "log.csv")
        secs = float(argv[2]) if len(argv) > 2 else 30.0
        print("wrote", log_csv(path, seconds=secs))
    elif cmd == "ble":
        secs = float(argv[1]) if len(argv) > 1 else 15.0
        for ln in ble_capture(seconds=secs):
            print(ln)
    elif cmd == "blelog":
        path = argv[1] if len(argv) > 1 else str(DATA_DIR / "ble_log.csv")
        secs = float(argv[2]) if len(argv) > 2 else 60.0
        print("wrote", ble_log_csv(path, seconds=secs, header=BLE_CSV_HEADER_LSI))
    elif cmd in ("selftest", "self_test"):
        rep = self_test()
        for ln in rep.pop("lines", []):
            print(ln)
        print("---", rep)
    else:
        print(__doc__)


if __name__ == "__main__":
    _main(sys.argv[1:])
