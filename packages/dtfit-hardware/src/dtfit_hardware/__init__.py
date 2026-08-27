"""dtfit-hardware: the real-silicon rig, hardware twin of ``realtime_gps``.

The ``realtime_gps`` domain in dtfit-experimental simulates a 9-DOF
GPS/inertial rig in NumPy. This package runs the same study on an Arduino
Nano 33 BLE Sense (onboard IMU and BLE) reading a NEO-M8N GPS.

* ``backend.py`` drives the board from the host: locate it, flash a sketch
  from ``firmware/``, capture the USB or BLE stream.
* ``compare_real.py`` scores captured logs against the simulation's
  ``realtime_gps.backend`` baselines: Kalman/CT-EKF, IMU fusion,
  glitch and float32.
* ``firmware/`` holds the Arduino sketches. ``nano_lsi_log`` is the one the
  rig actually runs.
* ``tools/`` holds the host tools; ``embed_lsi.py`` bakes the LSI tables
  into C.
* ``mobile/`` is a React Native app reading the ``dtfit-gps`` BLE telemetry
  live (built with yarn; see ``mobile/dtfit-monitor/README.md``).

``realtime_gps_hw.ipynb`` reproduces the simulation's E1-E7 story on real
silicon. Parts and wiring live in ``papers/embedded_hardware_bom.md``,
bring-up in ``papers/embedded_nano_build_guide.md``.
"""
