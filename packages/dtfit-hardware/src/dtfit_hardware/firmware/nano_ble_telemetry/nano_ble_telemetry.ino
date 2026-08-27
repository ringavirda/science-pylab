/*
 * nano_ble_telemetry - wireless telemetry for the realtime_gps_hw rig.
 *
 * Streams GPS and IMU telemetry over BLE, letting the rig run untethered on
 * battery while the host receives the log over Bluetooth. Each line is also
 * echoed to USB serial, so it stays readable when tethered.
 *
 * Advertises as "dtfit-gps" with one notify characteristic carrying a CSV line:
 *   t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz
 *
 * Wiring:  GPS TX -> D0 (RX0), GND shared, VCC on the 5 V rail; IMU onboard.
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: ArduinoBLE, TinyGPSPlus, Arduino_BMI270_BMM150
 */
#include <ArduinoBLE.h>
#include <TinyGPS++.h>
#include "Arduino_BMI270_BMM150.h"

#define SVC_UUID  "9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10"
#define TELE_UUID "9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10"

BLEService telemetry(SVC_UUID);
BLEStringCharacteristic teleChar(TELE_UUID, BLERead | BLENotify, 160);

TinyGPSPlus gps;

static String buildLine() {
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable())    IMU.readGyroscope(gx, gy, gz);
  int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
  int fix = gps.location.isValid() ? 1 : 0;

  String s = String(millis());
  s += ","; s += sats;
  s += ","; s += fix;
  s += ","; s += String(gps.location.lat(), 6);
  s += ","; s += String(gps.location.lng(), 6);
  s += ","; s += String(gps.altitude.meters(), 1);
  s += ","; s += String(gps.hdop.hdop(), 1);
  s += ","; s += String(gps.speed.kmph(), 1);
  s += ","; s += String(ax, 3); s += ","; s += String(ay, 3);
  s += ","; s += String(az, 3);
  s += ","; s += String(gx, 2); s += ","; s += String(gy, 2);
  s += ","; s += String(gz, 2);
  return s;
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  IMU.begin();

  if (!BLE.begin()) {
    Serial.println("BLE init FAILED");
    while (1) { }
  }
  BLE.setDeviceName("dtfit-gps");
  BLE.setLocalName("dtfit-gps");
  BLE.setAdvertisedService(telemetry);
  telemetry.addCharacteristic(teleChar);
  BLE.addService(telemetry);
  teleChar.writeValue("boot");
  BLE.advertise();
  Serial.println("BLE advertising as 'dtfit-gps'");
}

void loop() {
  static unsigned long last = 0;
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("BLE connected: ");
    Serial.println(central.address());
    while (central.connected()) {
      while (Serial1.available()) gps.encode(Serial1.read());
      if (millis() - last >= 1000) {
        last = millis();
        String line = buildLine();
        teleChar.writeValue(line);
        Serial.println(line);
      }
    }
    Serial.println("BLE disconnected");
    BLE.advertise();  // resume advertising, or the host cannot reconnect
  } else {
    while (Serial1.available()) gps.encode(Serial1.read());
    if (millis() - last >= 1000) {
      last = millis();
      Serial.println(buildLine());  // echo over USB while not connected
    }
  }
}
