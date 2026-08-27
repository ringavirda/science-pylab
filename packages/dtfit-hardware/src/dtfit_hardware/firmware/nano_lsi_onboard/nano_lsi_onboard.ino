/*
 * nano_lsi_onboard - the dtfit streaming LSI filter running on the M4F.
 *
 * A fixed-size float32 port of dtfit.streaming.LSIFilter (dtfit_lsi.h, tables
 * from tools/embed_lsi.py) on the Cortex-M4F. At boot it replays an embedded
 * real-data vector (lsi_testvec.h) through the filter and dumps each estimate
 * for the host to check against the float64 golden, reports the per-update
 * cost in CPU cycles and microseconds from the DWT cycle counter along with
 * the SRAM footprint, then runs live on the GPS longitude stream, printing the
 * filtered value and the forecast.
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: TinyGPSPlus
 * Monitor: 115200 baud
 */
#include "dtfit_lsi.h"
#include "lsi_testvec.h"
#include <TinyGPS++.h>

// DWT cycle counter. Present on the Cortex-M4, absent on an M0+.
static inline void dwtEnable() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwtCycles() { return DWT->CYCCNT; }

LsiFilter filt;
TinyGPSPlus gps;
unsigned long gStart = 0;

static void runValidation() {
  Serial.println();
  Serial.println("=== LSI on-MCU validation (float32) ===");
  Serial.print("config: W="); Serial.print(LSI_W);
  Serial.print(" order="); Serial.print(LSI_ORDER);
  Serial.print(" params="); Serial.print(LSI_N);
  Serial.print("  state footprint=");
  Serial.print((unsigned)sizeof(LsiFilter));
  Serial.println(" bytes");

  float p0[LSI_N];
  p0[0] = LSI_Y[0];
  for (int k = 1; k < LSI_N; k++) p0[k] = 0.0f;
  filt.reset(p0);

  uint32_t cycSum = 0, cycMax = 0;
  int nUpd = 0;
  for (int i = 0; i < LSI_NT; i++) {
    uint32_t c0 = dwtCycles();
    bool upd = filt.update(LSI_T[i], LSI_Y[i]);
    uint32_t dc = dwtCycles() - c0;
    if (upd) {
      cycSum += dc;
      if (dc > cycMax) cycMax = dc;
      nUpd++;
      Serial.print("VAL ");
      Serial.print(i);
      Serial.print(' ');
      Serial.print(filt.p[0], 7);
      Serial.print(' ');
      Serial.println(filt.p[1], 9);
    }
  }
  Serial.print("updates=");
  Serial.println(nUpd);
  if (nUpd) {
    float perUs = (float)cycSum / nUpd / (LSI_F_CPU_HZ / 1e6f);
    float maxUs = (float)cycMax / (LSI_F_CPU_HZ / 1e6f);
    Serial.print("cost: ");
    Serial.print(cycSum / nUpd);
    Serial.print(" cyc/update avg, ");
    Serial.print(perUs, 2);
    Serial.print(" us avg, ");
    Serial.print(maxUs, 2);
    Serial.println(" us max");
  }
  Serial.println("=== end validation ===");
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 12000) { }  // wait for the host capture
  Serial1.begin(9600);
  dwtEnable();

  runValidation();

  float p0[LSI_N] = {0.0f};
  filt.reset(p0);
  Serial.println("live: GPS longitude -> filtered + 1 s forecast");
}

void loop() {
  while (Serial1.available()) gps.encode(Serial1.read());

  static unsigned long last = 0;
  if (millis() - last < 1000) return;
  last = millis();

  if (!gps.location.isValid()) {
    Serial.println("live: (no GPS fix)");
    return;
  }
  if (gStart == 0) gStart = millis();
  float t = (millis() - gStart) / 1000.0f;
  float lon = (float)gps.location.lng();

  uint32_t c0 = dwtCycles();
  filt.update(t, lon);
  uint32_t dc = dwtCycles() - c0;

  Serial.print("t=");
  Serial.print(t, 1);
  Serial.print(" lon=");
  Serial.print(lon, 6);
  Serial.print(" filt=");
  Serial.print(filt.predict(t), 6);
  Serial.print(" fcast+1s=");
  Serial.print(filt.predict(t + 1.0f), 6);
  Serial.print(" cost=");
  Serial.print((float)dc / (LSI_F_CPU_HZ / 1e6f), 2);
  Serial.println("us");
}
