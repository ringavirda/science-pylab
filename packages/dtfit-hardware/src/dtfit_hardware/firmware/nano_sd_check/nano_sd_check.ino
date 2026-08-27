/*
 * nano_sd_check - verify the SPI microSD module (SdFat, exFAT-capable).
 *
 * Built on SdFat's SdFs (FAT16/FAT32/exFAT), so a stock 128 GB exFAT card
 * works unreformatted. It mounts the card, writes a test file, reads it back,
 * and reports the filesystem type plus any low-level card error code.
 *
 * Wiring:  VCC->3V3 or 5V (see note), GND, SCK->D13, MISO->D12, MOSI->D11, CS->D10.
 * Note:    modules with an onboard regulator + level-shifter want VCC = 5 V;
 *          bare 3.3 V adapters want VCC = 3V3.
 * Board:   Arduino Nano 33 BLE Sense Rev2.  Library: SdFat.  Monitor: 115200.
 */
#include <SPI.h>
#include "SdFat.h"

#define SD_CS 10
SdFs sd;
FsFile file;

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 12000) { }

  Serial.println();
  Serial.println("=== nano_sd_check (SdFat / exFAT-capable) ===");
  Serial.print("sd.begin(CS=D10, 4 MHz) ... ");
  // 4 MHz SPI, kept conservative for breadboard jumper wiring.
  if (!sd.begin(SdSpiConfig(SD_CS, SHARED_SPI, SD_SCK_MHZ(4)))) {
    Serial.println("FAILED");
    Serial.print("card errorCode=0x");
    Serial.print(sd.sdErrorCode(), HEX);
    Serial.print(" data=0x");
    Serial.println(sd.sdErrorData(), HEX);
    Serial.println("check: CS=D10 MOSI=D11 MISO=D12 SCK=D13, VCC, GND, card seated");
    return;
  }
  Serial.println("OK -- card mounted");
  Serial.print("FS type (32=FAT32, 64=exFAT): ");
  Serial.println(sd.fatType());

  sd.remove("rigtest.txt");
  file = sd.open("rigtest.txt", O_WRITE | O_CREAT | O_TRUNC);
  if (!file) { Serial.println("open-for-write FAILED"); return; }
  file.println("realtime_gps_hw SD test (SdFat)");
  file.println("line 2");
  file.close();
  Serial.println("wrote rigtest.txt");

  file = sd.open("rigtest.txt", O_READ);
  if (!file) { Serial.println("open-for-read FAILED"); return; }
  Serial.print("size: ");
  Serial.print(file.size());
  Serial.println(" bytes");
  Serial.println("read back:");
  while (file.available()) Serial.write(file.read());
  file.close();
  Serial.println("=== SD OK ===");
}

void loop() { }
