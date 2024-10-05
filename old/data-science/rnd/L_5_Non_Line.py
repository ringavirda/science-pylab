# ---------------- Лекція 4 нелінійне МНК згладжування за наявності АВ --------------

import numpy as np
import math as mt
import matplotlib.pyplot as plt

global Yout_NL

# -------------------------- сегмент API ---------------------------------
n=50; iter=int(n)                           # кількість вимірів експериментальної вибірки     -----
koef = 50                                   # інтервал прогнозу за кількісттю вимірів         -----
nAVv=10; nAV=int ((iter*nAVv)/100)           # кількість АВ у відсотках та абсолютних одиницях -----
dm=0; dsig=0.5                              # параметри нормального закону розподілу ВВ: середне та СКВ
SAV=np.zeros((nAV))
SSAV=np.zeros((nAV))

# -------------------------- МОДЕЛЬ ВИМІРІВ ------------------------------
# -------------- генерація номерів АВ за рівномірним законом  -----------
for i in range(nAV):
    SAV[i]=mt.ceil(np.random.randint(1, iter)) # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
print('номери АВ: SAV=',SAV)
# --------------------- нормальний закон розводілу ВВ ---------------------
S = np.random.normal(dm, dsig, iter)      # нормальний закон розподілу ВВ з вибіркою обємом iter та параметрами: dm, dsig
mS=np.median(S)
dS=np.var(S)
scvS=mt.sqrt(dS)

# ---------------- модель виміру (періодичний закон) з нормальний шумом -----------
SV=np.zeros((n)); S0=np.zeros((n)); SV0=np.zeros((n)); SV_AV=np.zeros((n))
for i in range(n):
    S0[i]=(7.8*mt.cos(0.05*i)+9.5*mt.sin(0.05*i))                # періодична модель реального процесу
    SV[i] = S0[i]+S[i]
    SV0[i] = abs(SV[i] - S0[i])              # урахування тренду в оцінках статистичних хараткеристик
    SV_AV[i] = SV[i]

# ----- модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
SSAV=np.random.normal(dm, (3*dsig), nAV)     # аномальна випадкова похибка з нормальним законом
for i in range(nAV):
    k=int (SAV[i])
    SV_AV[k] = S0[k] + SSAV[i]               # аномальні вимірів з рівномірно розподіленими номерами

# ---------------------- графіки тренда, вимірів  аномаліями ------------------
# plt.plot(SV0)
plt.plot(SV_AV)
plt.plot(S0)
plt.ylabel('динаміка продажів')
plt.show()
# -------------  статистичні характеристики трендової вибірки (з урахуванням тренду)
mSV0=np.mean(SV0)
dSV0=np.var(SV0)
scvSV0=mt.sqrt(dSV0)
print('-------- статистичны характеристики виміряної вибірки без АВ ----------')
print('матиматичне сподівання ВВ3=', mSV0)
print('дисперсія ВВ3 =', dSV0)
print('СКВ ВВ3=', scvSV0)

SV_AV0=np.zeros((n))
for i in range(n):
     SV_AV0[i] = abs(SV_AV[i] - S0[i])  # урахування тренду в оцінках статистичних хараткеристик

print('-- статистичны характеристики виміряної вибірки за НАЯВНОСТІ АВ -------')
mSV_AS=np.mean(SV_AV0)
dSV_AV=np.var(SV_AV)
scvSV_AV=mt.sqrt(dSV_AV)
print('матиматичне сподівання ВВ3=', mSV_AS)
print('дисперсія ВВ3 =', dSV_AV)
print('СКВ ВВ3=', scvSV_AV)
print('----------------------------------------------------------------------')
# ------------------------------ МНК згладжування -------------------------------------
Yin=np.zeros((iter, 1)); F=np.ones((iter, 3)); Yout_NL=np.zeros((iter, 1))
for i in range(iter):                          # формування структури вхідних матриць МНК
    Yin[i, 0] = float(S0[i])                   # формування матриці вхідних даних без аномілій
    F[i, 1] = float(i); F[i, 2] = float(i*i)   # формування матриці вхідних даних без аномілій
# ---------------------- функція обчислень алгоритму - MNK -----------------------------

def MNK (Yin, F):
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

# ---------------------- функція обчислень алгоритму нелінійного - MNK ----------------
def MNK_NL (Yin, F):
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    a0=C[0, 0]
    b0=C[1, 0]/(mt.sqrt(abs(C[2, 0]/C[0, 0])))
    w0 = mt.sqrt(abs(C[2, 0] / C[0, 0]))
    for i in range(iter):           # нелінійна крива МНК
        Yout_NL[i, 0] = (a0*mt.cos(w0*i)+b0*mt.sin(w0*i))
    return Yout_NL

# ---------------------- функція прогнозування за різними моделями - MNK ----------------
def MNK_NL_Extrapol (Yin, F):
    YReal = np.zeros(((iter + koef), 1))
    YMNK = np.zeros(((iter + koef), 1))
    YMNK_NL = np.zeros(((iter + koef), 1))
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    print('C[0, 0]=', C[0, 0])
    print('C[1, 0]=', C[1, 0])
    print('C[2, 0]=', C[2, 0])
    a0=C[0, 0]
    b0=C[1, 0]/(mt.sqrt(abs(C[2, 0]/C[0, 0])))
    w0 = mt.sqrt(abs(C[2, 0] / C[0, 0]))
    print('a0=', a0)
    print('b0=', b0)
    print('w0=', w0)
    for i in range(iter+koef):
        YReal[i, 0] = (7.8 * mt.cos(0.05 * i) + 9.5 * mt.sin(0.05 * i))    # ідеальна крива - вхідна
        YMNK[i, 0] = C[0, 0]+C[1, 0]*i+(C[2, 0]*i*i)                       # проліноміальна крива МНК
        YMNK_NL[i, 0] = (a0*mt.cos(w0*i)+b0*mt.sin(w0*i))                  # нелінійна крива МНК
    plt.plot(Yin)
    plt.plot(YReal)
    plt.plot(YMNK)
    plt.plot(YMNK_NL)
    plt.ylabel('динаміка продажів')
    plt.show()
    return

# ---------------------- застосування МНК до незашумлених вимірів -----------------------
Yout0 = MNK (Yin, F)
# ---------------------- застосування МНК до нормальних вимірів -----------------------
for i in range(iter):
    Yin[i, 0] = float(SV[i])
Yout1 = MNK (Yin, F)
# ---------------------- застосування МНК до аномальних вимірів -----------------------
for i in range(iter):
    Yin[i, 0] = float(SV_AV[i])
Yout2 = MNK (Yin, F)      # проліноміальна крива МНК
Yout3 = MNK_NL (Yin, F)   # нелінійна крива МНК
# ------------ графіки тренда, МНК оцінок нормального та аномального шуму ---------------
Yout00=np.zeros((n)); Yout10=np.zeros((n)); Yout20=np.zeros((n));
for i in range(n):
     Yout00[i] = abs(Yout0[i] - S0[i])
     Yout10[i] = abs(Yout1[i] - S0[i])
     Yout20[i] = abs(Yout2[i] - S0[i])

print('----------------------- статистичны характеристики виміряної вибірки за НАЯВНОСТІ АВ -----------------')
mYout00=np.mean(Yout00);  mYout10=np.mean(Yout10);  mYout20=np.mean(Yout20)
dYout00=np.var(Yout00);     dYout10=np.var(Yout10);     dYout20=np.var(Yout20)
scvYout00=mt.sqrt(dYout00); scvYout10=mt.sqrt(dYout10); scvYout20=mt.sqrt(dYout20)

print('--------------------------- за відсутності похибок ----- похибки нормальні ------- похибки аномальні ---')
print('матиматичне сподівання ВВ3=', mYout00 ,   '----', mYout10,  '----', mYout20)
print('дисперсія ВВ3 =            ', dYout00  ,  '----', dYout10,  '----', dYout20)
print('СКВ ВВ3=                   ', scvYout00  ,'----', scvYout10,'----', scvYout20)
print('-------------------------------------------------------------------------------------------------------')

plt.plot(Yin)
plt.plot(S0)
plt.plot(Yout2)
plt.plot(Yout3)
plt.ylabel('динаміка продажів')
plt.show()
# -------------------------- виклик функції екстраполяції за різними алгоритмами МНК ---------------------------
MNK_NL_Extrapol (Yin, F)
