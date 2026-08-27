/*
 * nano_rig_check - wiring verification for the realtime_gps_hw rig.
 *
 * A running health report for the wired-up rig. Each second it prints:
 *   - the external I2C scan (expect the INA226 at 0x40)
 *   - INA226 bus voltage / shunt voltage / current  (power-meter wiring)
 *   - onboard IMU acceleration                       (board sanity)
 *   - bytes/sec arriving on Serial1 (D0/D1)          (GPS UART link)
 *
 * The GPS must feed D0, which is Serial1 RX. On any other pin it reads
 * 0 bytes/s.
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: Arduino_BMI270_BMM150, INA226 (Rob Tillaart)
 * Monitor: 115200 baud
 */
#include <Wire.h>
#include "Arduino_BMI270_BMM150.h"
#include "INA226.h"

INA226 ina(0x40);
bool inaOk = false;

static void i2cScan() {
  byte found = 0;
  Serial.print("I2C (A4/A5):");
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print(" 0x");
      if (addr < 16) Serial.print('0');
      Serial.print(addr, HEX);
      found++;
    }
  }
  if (!found) Serial.print(" (none)");
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) { }

  Serial.println();
  Serial.println("=== nano_rig_check : realtime_gps_hw ===");

  Serial1.begin(9600);   // NEO-M8N UART on D0(RX)/D1(TX)
  Wire.begin();
  i2cScan();

  inaOk = ina.begin();
  if (inaOk) ina.setMaxCurrentShunt(0.5, 0.1);  // 0.1 ohm onboard shunt
  Serial.println(inaOk ? "INA226: found at 0x40" : "INA226: NOT found");

  if (!IMU.begin()) Serial.println("IMU: INIT FAILED");
}

void loop() {
  static unsigned long last = 0;
  static int gpsBytes = 0;
  static char sample[40];
  static int si = 0;

  while (Serial1.available()) {
    char c = Serial1.read();
    gpsBytes++;
    if (si < 39 && c >= 32 && c < 127) sample[si++] = c;
  }

  if (millis() - last < 1000) return;
  last = millis();
  sample[si] = '\0';

  if (!inaOk) {                  // live re-detect as you fix the I2C wiring
    inaOk = ina.begin();
    if (inaOk) ina.setMaxCurrentShunt(0.5, 0.1);
  }
  i2cScan();                     // show what's on A4/A5 each second

  Serial.print("INA226 ");
  if (inaOk) {
    Serial.print(ina.getBusVoltage(), 3);
    Serial.print(" Vbus  ");
    Serial.print(ina.getShuntVoltage_mV(), 3);
    Serial.print(" mVshunt  ");
    Serial.print(ina.getCurrent_mA(), 1);
    Serial.print(" mA");
  } else {
    Serial.print("(absent)");
  }

  float ax = 0, ay = 0, az = 0;
  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  Serial.print("  | IMU a=");
  Serial.print(ax, 2); Serial.print(',');
  Serial.print(ay, 2); Serial.print(','); Serial.print(az, 2);

  Serial.print("  | GPS bytes/s=");
  Serial.print(gpsBytes);
  if (gpsBytes > 0) {
    Serial.print(" \"");
    Serial.print(sample);
    Serial.print("\"");
  }
  Serial.println();

  gpsBytes = 0;
  si = 0;
}
