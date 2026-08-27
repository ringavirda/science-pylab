/*
 * nano_lsi_log - the rig's working firmware. The dtfit LSI runs on the MCU
 * against the live GPS, and the raw GPS+IMU stream and the on-board filter
 * estimate both go out over BLE and onto the SD card at once, with no field
 * reflash. The host then scores the float32 on-MCU result against its float64
 * reference.
 *
 * Coordinates. The fix becomes local east/north metres about the first-fix
 * origin before it is filtered. Raw degrees around 50/30 sit at the float32
 * floor and cost the few metres of precision we measured; small metre values
 * do not. The estimate returns to lat/lon only for the log line.
 *
 * Model. The order is switched from the inertial readings, which is the dtfit
 * idea that the filter carries the model and the model adapts to the regime.
 * Still, meaning gyro and accel at rest with GPS speed near zero, applies a
 * zero-velocity update: the degree-1 model collapses to degree-0, a position
 * average, and the forecast flattens instead of extrapolating drift as motion.
 * Moving runs the full degree-1 c0 + c1*t and tracks velocity. Same filter,
 * same tables; the switch is a constraint on the public state rather than a
 * second method, and the ``mode`` column records which one is active.
 *
 * GPS rate. configureGps() sends UBX to the NEO-M8N at boot, trimming NMEA to
 * RMC+GGA and setting 5 Hz (200 ms). That fits inside 9600 baud, so the baud
 * rate is deliberately left alone: a warm reset restarts the Nano but not the
 * GPS, and a changed baud would not survive it. At 5 Hz the fixed 15-sample
 * LSI window spans 3 s rather than 15, which tracks a fast vehicle far better.
 *
 * Heading. One gyro sample per second aliases turns badly enough to leave the
 * host heading near random, hence the high-rate gravity-aligned integrator in
 * integrateHeading() and the GPS-course anchor in anchorHeading().
 * ``hdg_deg`` carries the anchored, drift-free heading and ``dhdg_deg`` the
 * raw gyro increment, which is the one the host dead-reckons on.
 *
 * Recording is a phone toggle over the CTRL characteristic and starts off, so
 * only real drives get a file: no garage or walking noise, and one drive is
 * one self-bounded file. Telemetry streams over BLE whatever the state.
 *
 * Record (28 cols):
 *   t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz,mx,my,mz,
 *   est_lat,est_lon,fc_lat,fc_lon,mode,cost_us,hdg_deg,dhdg_deg,imu_hz,newfix,sd
 *   ``imu_hz`` is the IMU samples integrated this interval, which confirms the
 *   rate. ``newfix`` is 1 when the line carries a fresh GPS fix and 0 for a
 *   between-fix sample. ``sd`` is a 3-state, always last, that the phone
 *   renders as OFF / REC / NO SD: 0 paused by the user, 1 recording with the
 *   card OK, 2 recording but the SD failing. Tapping the chip toggles it.
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: ArduinoBLE, TinyGPSPlus, Arduino_BMI270_BMM150, SdFat
 * Monitor: 115200 baud
 */
#include <ArduinoBLE.h>
#include <TinyGPS++.h>
#include "Arduino_BMI270_BMM150.h"
#include <SPI.h>
#include "SdFat.h"
#include "dtfit_lsi.h"
#include <math.h>

#define SVC_UUID  "9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10"
#define TELE_UUID "9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10"
#define CTRL_UUID "9a1e0002-1b2c-4f3a-8d5e-6f7a8b9c0d10"   // phone -> board: "1" record, "0" pause
BLEService telemetry(SVC_UUID);
BLEStringCharacteristic teleChar(TELE_UUID, BLERead | BLENotify, 200);
BLEStringCharacteristic ctrlChar(CTRL_UUID, BLEWrite, 8);

#define SD_CS 10
#define REC_DEFAULT false       // start paused; only real drives get a file
SdFs sd;
FsFile sdLog;
bool sdOk = false;
bool sdWriting = false;      // did the last write+sync to the card succeed?
bool recording = REC_DEFAULT; // each enable opens a file, disable closes it

// Emit/log/BLE rate, matched to the GPS rate to log every fix. Emitting
// barely blocks now that emit() batches the SD sync(), leaving the loop
// close to free for the high-rate IMU integrator. 5 notify/s sits well
// under the ~20-30/s BLE ceiling for this characteristic.
#define EMIT_HZ 5
#define SYNC_MS 300         // batch the SD sync() to ~3 Hz (power-loss window <~0.3 s)

static const char* HEADER =
  "t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz,mx,my,mz,"
  "est_lat,est_lon,fc_lat,fc_lon,mode,cost_us,hdg_deg,dhdg_deg,imu_hz,newfix,sd";

static const double M_PER_DEG = 111320.0;   // metres per degree of latitude

TinyGPSPlus gps;

// Per-session logging: enabling recording opens a fresh timestamped file,
// never appending to or rewriting an earlier one.
char logName[24] = {0};      // current SD file name for this session
bool logNamed = false;       // renamed from SESSnnnn to the GPS-UTC name yet?

// FAT modify-time from GPS UTC, giving each file a real timestamp the host
// can sort on. Falls back to a fixed date before the first fix.
static void gpsFatDateTime(uint16_t* date, uint16_t* time) {
  if (gps.date.isValid() && gps.time.isValid() && gps.date.year() >= 2021) {
    *date = FS_DATE(gps.date.year(), gps.date.month(), gps.date.day());
    *time = FS_TIME(gps.time.hour(), gps.time.minute(), gps.time.second());
  } else {
    *date = FS_DATE(2026, 1, 1);
    *time = FS_TIME(0, 0, 0);
  }
}

// Once GPS UTC is valid, rename this session's file from its SESSnnnn
// placeholder to YYYYMMDD_HHMMSS.CSV. Done once and early, before real
// motion, leaving the run to append to a stable open file. With no fix it
// stays SESSnnnn, still unique and with no data lost.
static void maybeNameLog() {
  if (!sdOk || logNamed || !sdLog) return;
  if (!(gps.date.isValid() && gps.time.isValid() && gps.date.year() >= 2021)) return;
  char ts[24];
  snprintf(ts, sizeof(ts), "%04u%02u%02u_%02u%02u%02u.CSV",
           gps.date.year(), gps.date.month(), gps.date.day(),
           gps.time.hour(), gps.time.minute(), gps.time.second());
  logNamed = true;                       // attempt once regardless of outcome
  if (sd.exists(ts)) return;             // same-second file: keep SESSnnnn
  if (sdLog.rename(ts)) {
    strncpy(logName, ts, sizeof(logName) - 1);
    sdLog.sync();
    Serial.print("SD: session file -> "); Serial.println(logName);
  }
}

LsiFilter fE, fN;            // degree-1 LSI on local East / North metres
bool   originSet = false;
double lat0 = 0, lon0 = 0;   // ENU origin (first fix)
double cosLat0 = 1.0;
unsigned long gStart = 0;
int  staticCount = 0;        // debounce for the rest detector
bool prevStatic = false;

// High-rate IMU and gravity-aligned heading integration; runs every loop.
float imuAx = 0, imuAy = 0, imuAz = 0;   // latest accel (g), cached for the log line
float imuGx = 0, imuGy = 0, imuGz = 0;   // latest gyro (deg/s)
double hdgPsi = 0.0;         // raw integrated heading (rad) about gravity
double hdgPsiEmit = 0.0;     // raw heading at the previous emit
double hdgClean = 0.0;       // gyro plus the slow GPS-course anchor (rad)
float  yawBias = 0.0;        // gravity-vertical yaw-rate bias (deg/s), ZUPT-adapted
float  gravf[3] = {0, 0, 0}; // low-passed accel = gravity direction estimate (g)
bool   gravInit = false;
unsigned long lastImuUs = 0;
uint16_t imuCount = 0;       // IMU samples integrated since the last emit

// The complementary GPS-course anchor for hdgClean. A causal
// backward-difference course over the last CRS_WIN fixes, taken only where
// displacement and motion make it real, nudges the gyro heading toward it.
// hdgClean therefore tracks turns on the gyro yet cannot drift. The
// constants were tuned on the real drive, 82 down to ~15 deg RMS against
// GPS course. CRS_N exceeds CRS_WIN+1 to keep the whole window in the ring.
#define CRS_WIN  15          // look-back fixes for the course (~3 s at 5 Hz)
#define CRS_DISP 8.0f        // min displacement (m) over that window for a valid course
#define CRS_K    0.04f       // correction gain per fix toward the GPS course (matches the host)
#define CRS_N    20          // ring capacity
float crsE[CRS_N] = {0}, crsN[CRS_N] = {0};   // ring of recent fix ENU (east/north metres)
int   crsHead = 0, crsFill = 0;

static inline void dwtEnable() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwtCycles() { return DWT->CYCCNT; }

// Zero-velocity update: pin the velocity parameter to 0 and decouple its
// covariance, leaving the degree-1 model to behave as a constant while still.
static void zupt(LsiFilter& f) {
  f.p[1] = 0.0f;
  f.P[0][1] = 0.0f; f.P[1][0] = 0.0f; f.P[1][1] = 1e-6f;
}
// Release the velocity by re-inflating its covariance when motion resumes,
// or the filter stays pinned near zero instead of re-acquiring quickly.
static void release(LsiFilter& f) {
  f.P[1][1] = LSI_P0_DIAG;
  f.P[0][1] = 0.0f; f.P[1][0] = 0.0f;
}

// Poll the IMU as fast as it has data and integrate the gravity-aligned yaw
// rate into an absolute heading. At a few us per sample it can run every
// loop, giving ~100 Hz and killing the 1 Hz aliasing that made the host
// heading useless. Gravity is a low-passed accel, which keeps the yaw axis
// on board tilt rather than leaking roll and pitch into "yaw" as raw gyro_z
// did, for 98 deg of error. A rest-gated bias, on the same rest test the LSI
// mode uses, stops the heading drifting while parked.
static void integrateHeading() {
  if (!(IMU.gyroscopeAvailable() && IMU.accelerationAvailable())) return;
  float gx, gy, gz, ax, ay, az;
  IMU.readGyroscope(gx, gy, gz);          // deg/s
  IMU.readAcceleration(ax, ay, az);       // g
  imuGx = gx; imuGy = gy; imuGz = gz;     // cache the latest sample for the log line
  imuAx = ax; imuAy = ay; imuAz = az;
  unsigned long now = micros();
  float dt = lastImuUs ? (now - lastImuUs) * 1e-6f : 0.0f;
  lastImuUs = now;
  if (dt <= 0.0f || dt > 0.2f) return;    // ignore the first sample / long stalls
  if (!gravInit) { gravf[0] = ax; gravf[1] = ay; gravf[2] = az; gravInit = true; }
  float b = dt / (0.5f + dt);             // ~0.5 s gravity low-pass (tracks slow tilt only)
  gravf[0] += b * (ax - gravf[0]);
  gravf[1] += b * (ay - gravf[1]);
  gravf[2] += b * (az - gravf[2]);
  float gn = sqrtf(gravf[0] * gravf[0] + gravf[1] * gravf[1] + gravf[2] * gravf[2]);
  if (gn < 1e-3f) return;
  float yr = (gx * gravf[0] + gy * gravf[1] + gz * gravf[2]) / gn;   // deg/s about gravity
  float gm = sqrtf(gx * gx + gy * gy + gz * gz);
  float am = sqrtf(ax * ax + ay * ay + az * az);
  float spd = gps.location.isValid() ? gps.speed.kmph() : 0.0f;
  if (gm < 3.0f && fabsf(am - 1.0f) < 0.05f && spd < 2.0f)
    yawBias = 0.995f * yawBias + 0.005f * yr;                        // ZUPT the yaw-rate bias
  double incr = (double)(yr - yawBias) * dt * (M_PI / 180.0);
  hdgPsi   += incr;          // raw integrator; feeds the dhdg_deg column
  hdgClean += incr;          // same increment, anchored to GPS course below
  imuCount++;
}

// On each fresh fix, nudge hdgClean toward the causal GPS course, the
// direction from the fix CRS_WIN samples ago to now, when there is real
// displacement and motion. The gyro gives the fast response and this slow
// pull takes out the drift and the loose-mount vibration bias. The push
// happens after the correction, so ``old`` indexes the fix exactly CRS_WIN
// steps back.
static void anchorHeading(float e, float n, bool moving) {
  if (crsFill >= CRS_WIN && moving) {
    int old = (crsHead - CRS_WIN + CRS_N) % CRS_N;
    float de = e - crsE[old], dn = n - crsN[old];
    if (sqrtf(de * de + dn * dn) > CRS_DISP) {
      float course = atan2f(dn, de);
      float d = atan2f(sinf(course - (float)hdgClean), cosf(course - (float)hdgClean));
      hdgClean += (double)(CRS_K * d);
    }
  }
  crsE[crsHead] = e; crsN[crsHead] = n;
  crsHead = (crsHead + 1) % CRS_N;
  if (crsFill < CRS_N) crsFill++;
}

static String buildLine() {
  // Accel and gyro come from the high-rate integrator's latest sample, one
  // reader only, leaving nothing to contend with integrateHeading. The mag
  // is read here because the integrator has no use for it.
  float ax = imuAx, ay = imuAy, az = imuAz, gx = imuGx, gy = imuGy, gz = imuGz;
  float mx = 0, my = 0, mz = 0;
  if (IMU.magneticFieldAvailable()) IMU.readMagneticField(mx, my, mz);  // BMM150 (uT)
  int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
  int fix = gps.location.isValid() ? 1 : 0;
  bool newfix = gps.location.isUpdated();   // a fresh GPS position parsed since the last emit?

  // rest detector from the inertial readings, corroborated by GPS speed
  float gMag = sqrtf(gx * gx + gy * gy + gz * gz);          // deg/s
  float aMag = sqrtf(ax * ax + ay * ay + az * az);          // g (~1 at rest)
  float spd = fix ? gps.speed.kmph() : 0.0f;
  bool restNow = (gMag < 3.0f) && (fabsf(aMag - 1.0f) < 0.05f) && (spd < 2.0f);
  if (restNow) { if (staticCount < 5) staticCount++; } else staticCount = 0;
  bool isStatic = (staticCount >= 3);

  double estLat = 0, estLon = 0, fcLat = 0, fcLon = 0;
  float costUs = 0; int mode = isStatic ? 0 : 1;
  if (fix) {
    double lat = gps.location.lat(), lon = gps.location.lng();
    if (!originSet) {
      lat0 = lat; lon0 = lon; cosLat0 = cos(lat0 * M_PI / 180.0);
      float pe[LSI_N] = {0}, pn[LSI_N] = {0};   // origin -> ENU (0,0)
      fE.reset(pe); fN.reset(pn);
      gStart = millis(); originSet = true;
    }
    // local ENU metres about the origin; small enough for float32
    float e = (float)((lon - lon0) * cosLat0 * M_PER_DEG);
    float n = (float)((lat - lat0) * M_PER_DEG);
    float t = (millis() - gStart) / 1000.0f;

    if (newfix) {              // only a genuinely new GPS sample is ingested:
      // emitting faster than the GPS must not feed the window duplicate fixes.
      if (prevStatic && !isStatic) { release(fE); release(fN); }   // motion resumed
      uint32_t c0 = dwtCycles();
      fE.update(t, e);
      fN.update(t, n);
      if (isStatic) { zupt(fE); zupt(fN); }      // still -> degree-0 (ZUPT)
      uint32_t dc = dwtCycles() - c0;
      costUs = (float)dc / (LSI_F_CPU_HZ / 1e6f);
      prevStatic = isStatic;
      anchorHeading(e, n, spd >= 2.0f);          // slow-anchor to the course
    }

    float estE = fE.predict(t), estN = fN.predict(t);   // re-evaluated every emit (free)
    float fcE = isStatic ? estE : fE.predict(t + 1.0f);
    float fcN = isStatic ? estN : fN.predict(t + 1.0f);
    // back to lat/lon for the log; the large origin is added in double
    estLat = lat0 + (double)estN / M_PER_DEG;
    estLon = lon0 + (double)estE / (M_PER_DEG * cosLat0);
    fcLat  = lat0 + (double)fcN / M_PER_DEG;
    fcLon  = lon0 + (double)fcE / (M_PER_DEG * cosLat0);
  }

  String s = String(millis());
  s += ","; s += sats;
  s += ","; s += fix;
  s += ","; s += String(gps.location.lat(), 6);
  s += ","; s += String(gps.location.lng(), 6);
  s += ","; s += String(gps.altitude.meters(), 1);
  s += ","; s += String(gps.hdop.hdop(), 1);
  s += ","; s += String(gps.speed.kmph(), 1);
  s += ","; s += String(ax, 3); s += ","; s += String(ay, 3); s += ","; s += String(az, 3);
  s += ","; s += String(gx, 2); s += ","; s += String(gy, 2); s += ","; s += String(gz, 2);
  s += ","; s += String(mx, 1); s += ","; s += String(my, 1); s += ","; s += String(mz, 1);
  s += ","; s += String(estLat, 6); s += ","; s += String(estLon, 6);
  s += ","; s += String(fcLat, 6);  s += ","; s += String(fcLon, 6);
  s += ","; s += mode;
  s += ","; s += String(costUs, 1);
  // High-rate heading block, integrated between emits by integrateHeading.
  // hdg_deg is the cleaned, GPS-course-anchored heading; dhdg_deg is the raw
  // gyro increment, which is what the host dead-reckons and post-filters on.
  double hdgDeg = fmod(hdgClean * 180.0 / M_PI, 360.0);
  if (hdgDeg < 0) hdgDeg += 360.0;
  double dhdgDeg = (hdgPsi - hdgPsiEmit) * 180.0 / M_PI;   // RAW yaw change since the last emit
  hdgPsiEmit = hdgPsi;
  s += ","; s += String(hdgDeg, 1);
  s += ","; s += String(dhdgDeg, 2);
  s += ","; s += imuCount;   // IMU samples this interval (~rate/EMIT_HZ)
  s += ","; s += (newfix ? 1 : 0);
  imuCount = 0;
  // SD/recording state, always last: 0 paused, 1 recording, 2 no write
  int recState = !recording ? 0 : (sdWriting ? 1 : 2);
  s += ","; s += recState;
  return s;
}

// Write a UBX frame to the GPS: sync, class/id, length, payload, and the
// 8-bit Fletcher checksum over everything from the class byte on.
static void sendUBX(uint8_t cls, uint8_t id, const uint8_t* payload, uint16_t len) {
  uint8_t head[6] = {0xB5, 0x62, cls, id, (uint8_t)(len & 0xFF), (uint8_t)(len >> 8)};
  uint8_t ckA = 0, ckB = 0;
  for (int i = 2; i < 6; i++) { ckA += head[i]; ckB += ckA; }
  for (uint16_t i = 0; i < len; i++) { ckA += payload[i]; ckB += ckA; }
  Serial1.write(head, 6);
  if (len) Serial1.write(payload, len);
  Serial1.write(ckA); Serial1.write(ckB);
  Serial1.flush();
}
static void setMsgRate(uint8_t nmeaId, uint8_t rate) {   // class 0xF0 = standard NMEA
  uint8_t p[3] = {0xF0, nmeaId, rate};
  sendUBX(0x06, 0x01, p, 3);                             // UBX-CFG-MSG (current port)
  delay(20);
}
// Trim NMEA to RMC+GGA, all TinyGPSPlus needs, and set 5 Hz. That fits inside
// 9600 baud, leaving the baud rate alone and a warm reset unable to desync
// the link. Verified at 10 sentences/s, 5 Hz, about 43% of budget.
static void configureGps() {
  setMsgRate(0x00, 1);   // GGA on
  setMsgRate(0x04, 1);   // RMC on
  setMsgRate(0x01, 0);   // GLL off
  setMsgRate(0x02, 0);   // GSA off
  setMsgRate(0x03, 0);   // GSV off
  setMsgRate(0x05, 0);   // VTG off
  const uint8_t rate[6] = {0xC8, 0x00, 0x01, 0x00, 0x01, 0x00};  // measRate 200 ms, 1 cycle, GPS
  sendUBX(0x06, 0x08, rate, 6);                                  // UBX-CFG-RATE
  delay(20);
}

// Open a fresh per-session file when the phone enables recording: a new
// SESSnnnn.CSV, renamed to the GPS-UTC time on the first fix. The ENU origin,
// LSI and heading all reset, which keeps each file self-contained and keeps
// ENU metres small and float32-safe even when drives start far apart. Returns
// false with no card, and the phone then shows NO SD.
static bool openSession() {
  if (!sdOk) return false;
  if (sdLog) { sdLog.sync(); sdLog.close(); }     // close any previous session
  // Monotonic across enables this boot. Never reusing a name stops a fresh
  // enable O_TRUNCing a prior session, which may still be SESSnnnn if its
  // GPS-UTC rename has not happened yet.
  static int sessCounter = 0;
  do { snprintf(logName, sizeof(logName), "SESS%04d.CSV", sessCounter++); }
  while (sd.exists(logName) && sessCounter < 10000);
  sdLog = sd.open(logName, O_WRITE | O_CREAT | O_TRUNC);
  if (!sdLog) return false;
  sdLog.println(HEADER);
  sdLog.sync();
  logNamed = false;                    // maybeNameLog renames to GPS-UTC on the next fix
  originSet = false;                   // fresh ENU origin + LSI state for this file
  prevStatic = false; staticCount = 0;
  hdgPsi = 0.0; hdgPsiEmit = 0.0;      // raw heading restarts at 0; the host
                                       // fits the constant offset back
  hdgClean = 0.0; crsHead = 0; crsFill = 0;   // cleaned heading + ring restart
  Serial.print("SD: new session -> "); Serial.println(logName);
  return true;
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 8000) { }
  Serial1.begin(9600);
  delay(200);
  configureGps();          // trim NMEA + set 5 Hz (needs GPS RX<-Nano TX wired)
  IMU.begin();
  dwtEnable();

  sdOk = sd.begin(SdSpiConfig(SD_CS, SHARED_SPI, SD_SCK_MHZ(4)));
  if (sdOk) FsDateTime::setCallback(gpsFatDateTime);
  sdWriting = false;                 // no file until the phone says record
  Serial.println(sdOk ? "SD: ready (recording OFF -- enable from the phone to open a new file)"
                      : "SD: not present (BLE/serial only)");

  if (!BLE.begin()) {
    Serial.println("BLE init FAILED");
  } else {
    BLE.setDeviceName("dtfit-gps");
    BLE.setLocalName("dtfit-gps");
    BLE.setAdvertisedService(telemetry);
    telemetry.addCharacteristic(teleChar);
    telemetry.addCharacteristic(ctrlChar);   // phone writes "1"/"0" to record/pause
    BLE.addService(telemetry);
    teleChar.writeValue("boot");
    BLE.advertise();
    Serial.println("BLE advertising as 'dtfit-gps'");
  }
  Serial.println("model: ENU-LSI, IMU-adaptive (mode 0=still/ZUPT degree-0, 1=moving degree-1)");
  Serial.println(HEADER);
}

static void emit(const String& line) {
  Serial.println(line);
  if (recording && sdOk && sdLog) {
    sdLog.println(line);               // buffered write (fast); sync() is the slow, blocking part
    // Batch the sync to ~SYNC_MS. A per-line sync() blocks tens of ms and at a
    // 5 Hz emit would starve the high-rate heading integrator. The cost is
    // that a power loss now drops the unsynced tail, under SYNC_MS of it,
    // which is a fair trade for 5 Hz logging. Write health, and with it the
    // phone's REC / NO SD, is refreshed on each sync.
    static unsigned long lastSync = 0;
    if (millis() - lastSync >= SYNC_MS) {
      lastSync = millis();
      sdWriting = sdLog.sync() && !sdLog.getWriteError();
      if (!sdWriting) sdLog.clearWriteError();
    }
  } else if (recording) {
    sdWriting = false;                 // recording wanted but no card/file -> phone shows NO SD
  }
  // While paused the SD writes are simply skipped. The file itself is closed
  // by the DISABLE branch in loop(), and re-enabling opens a new one. A
  // parked or walking stretch therefore never lands in a drive's file.
  teleChar.writeValue(line);           // telemetry keeps streaming regardless of recording state
}

void loop() {
  static unsigned long last = 0;
  BLE.central();
  if (ctrlChar.written()) {                 // phone toggled recording: "1" record, "0" pause
    String v = ctrlChar.value();
    if (v.length()) {
      bool want = (v[0] != '0');
      if (want && !recording) {             // ENABLE -> open a fresh file (one drive = one file)
        recording = true;
        openSession();                      // no card -> emit() reports NO SD (sd col = 2)
      } else if (!want && recording) {      // DISABLE -> close the current file
        recording = false;
        if (sdLog) { sdLog.sync(); sdLog.close(); }
      }
      Serial.print("recording -> "); Serial.println(recording ? "ON" : "PAUSED");
    }
  }
  while (Serial1.available()) gps.encode(Serial1.read());
  integrateHeading();        // poll + integrate the IMU every iteration (~100 Hz)
  if (millis() - last >= (1000 / EMIT_HZ)) {
    last = millis();
    emit(buildLine());
    maybeNameLog();          // rename SESSnnnn -> GPS-UTC name once, at first fix
  }
}
