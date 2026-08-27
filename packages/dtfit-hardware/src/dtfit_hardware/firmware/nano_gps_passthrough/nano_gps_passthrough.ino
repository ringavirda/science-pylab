/*
 * nano_gps_passthrough - GPS bring-up for the realtime_gps_hw rig.
 *
 * Bridges the NEO-M8N UART (Serial1 on D0/D1) straight to USB, so you can
 * watch the raw NMEA sentences ($GxGGA / $GxRMC ...) arrive. This is the
 * wiring proof for the GPS link, before any parsing.
 *
 * Wiring:  GPS TX -> D0 (RX), GPS RX -> D1 (TX), shared GND, GPS VCC on the
 *          5 V rail. The M8N's default UART baud is 9600.
 * Monitor: 115200 baud (USB). Empty lat/lon until it has sky view + a fix.
 */
void setup() {
  Serial.begin(115200);   // USB to the host
  Serial1.begin(9600);    // NEO-M8N UART on D0(RX)/D1(TX)
}

void loop() {
  while (Serial1.available()) Serial.write(Serial1.read());  // GPS -> PC
  while (Serial.available())  Serial1.write(Serial.read());  // PC  -> GPS
}
