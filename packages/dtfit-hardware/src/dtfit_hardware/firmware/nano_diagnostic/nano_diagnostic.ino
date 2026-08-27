/*
 * nano_diagnostic - first bring-up step for the realtime_gps_hw rig.
 *
 * Proves the board with no external wiring at all: the USB upload path,
 * USB-serial telemetry, the onboard 9-axis IMU (BMI270 + BMM150) and the
 * external I2C bus on A4/A5. It streams IMU readings twice a second, which is
 * what the host backend watches to confirm the board is alive and the
 * toolchain works end to end.
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: Arduino_BMI270_BMM150
 * Monitor: 115200 baud
 */
#include <Wire.h>
#include "Arduino_BMI270_BMM150.h"

static void i2cScan() {
  byte found = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  I2C device at 0x");
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
      found++;
    }
  }
  Serial.print("  external I2C (A4/A5) devices: ");
  Serial.println(found);  // 0 until the INA226 is wired
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) { }  // wait up to 3 s for the host

  Serial.println();
  Serial.println("=== nano_diagnostic : realtime_gps_hw ===");
  Serial.println("board: Arduino Nano 33 BLE Sense Rev2");

  Wire.begin();
  Serial.println("scanning external I2C (A4/A5)...");
  i2cScan();

  if (!IMU.begin()) {
    Serial.println("IMU: INIT FAILED (need lib Arduino_BMI270_BMM150)");
  } else {
    Serial.print("IMU: OK  accel ");
    Serial.print(IMU.accelerationSampleRate());
    Serial.print(" Hz  gyro ");
    Serial.print(IMU.gyroscopeSampleRate());
    Serial.println(" Hz");
  }
  Serial.println("stream: IMU a=ax,ay,az(g) g=gx,gy,gz(dps) m=mx,my,mz(uT)");
}

void loop() {
  digitalWrite(LED_BUILTIN, (millis() / 500) % 2);  // 1 Hz heartbeat

  static unsigned long last = 0;
  if (millis() - last < 500) return;
  last = millis();

  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  float mx = 0, my = 0, mz = 0;
  if (IMU.accelerationAvailable())  IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable())     IMU.readGyroscope(gx, gy, gz);
  if (IMU.magneticFieldAvailable()) IMU.readMagneticField(mx, my, mz);

  Serial.print("IMU a=");
  Serial.print(ax, 2); Serial.print(',');
  Serial.print(ay, 2); Serial.print(',');
  Serial.print(az, 2);
  Serial.print("  g=");
  Serial.print(gx, 1); Serial.print(',');
  Serial.print(gy, 1); Serial.print(',');
  Serial.print(gz, 1);
  Serial.print("  m=");
  Serial.print(mx, 1); Serial.print(',');
  Serial.print(my, 1); Serial.print(',');
  Serial.println(mz, 1);
}
