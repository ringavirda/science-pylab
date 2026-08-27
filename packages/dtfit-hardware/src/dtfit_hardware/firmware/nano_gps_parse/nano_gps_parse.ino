/*
 * nano_gps_parse - GPS fix readout for the realtime_gps_hw rig.
 *
 * Parses the NEO-M8N's NMEA stream (Serial1 on D0/D1) with TinyGPS++ and
 * prints one clean line per second: satellites tracked, fix validity,
 * latitude, longitude, altitude, HDOP and speed. Run it once the raw
 * passthrough shows sentences arriving, to watch the fix come in.
 *
 * Wiring:  GPS TX -> D0 (RX0), GND shared, VCC on the 5 V rail.
 * Library: TinyGPSPlus
 * Monitor: 115200 baud (GPS UART is 9600).
 */
#include <TinyGPS++.h>

TinyGPSPlus gps;

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  Serial.println("nano_gps_parse: waiting for NMEA on Serial1 (D0)...");
}

void loop() {
  while (Serial1.available()) gps.encode(Serial1.read());

  static unsigned long last = 0;
  if (millis() - last < 1000) return;
  last = millis();

  Serial.print("sats=");
  Serial.print(gps.satellites.isValid() ? gps.satellites.value() : 0);
  Serial.print("  fix=");
  Serial.print(gps.location.isValid() ? "YES" : "no ");
  if (gps.location.isValid()) {
    Serial.print("  lat=");
    Serial.print(gps.location.lat(), 6);
    Serial.print(" lon=");
    Serial.print(gps.location.lng(), 6);
    Serial.print("  alt=");
    Serial.print(gps.altitude.meters(), 1);
    Serial.print("m  hdop=");
    Serial.print(gps.hdop.hdop(), 1);
    Serial.print("  spd=");
    Serial.print(gps.speed.kmph(), 1);
    Serial.print("km/h");
  }
  Serial.print("  (chars=");
  Serial.print(gps.charsProcessed());
  Serial.print(")");
  Serial.println();
}
