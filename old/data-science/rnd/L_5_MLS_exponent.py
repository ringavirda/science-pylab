# ----------------------------- Лекція 4 МНК експонента -------------------------------
# Завдання:
# Програма повинна забезпечувати МНК згладжування модельної статистичної вибірки з АВ
# Передбачити наступний функціонал скрипта:
# 1. Виявлення аномальних вимірів;
# 2. МНК прогнозування;
# 3. Кількість елементів вибірки може змінюватись;
# 4. Здійснити розрахунок статистичних характеристик випадкової величини;
# 5. Побудувати гістограму закону розподілу у формі графіку до та після згладжування;
# 6. Скрипт –  консольний (без графічного інтерфейсу).
# ---- підключення модулів (бібліотек)  Python методи яких буде використано в програмі ------
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -------------------------------- ФУНКЦІЇ МОДЕЛЬ ВИМІРІВ ----------------------------------
def Stat_characteristics(SL, Text):
    # статистичні характеристики ВВ з урахуванням тренду
    Yout = MNK(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print("------------", Text, "-------------")
    print("матиматичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    print("-----------------------------------------------------")
    # гістограма закону розподілу ВВ
    plt.hist(SL0, bins=20, facecolor="blue", alpha=0.5)
    plt.ylabel(Text)
    plt.show()
    return


# ----------- рівномірний закон розводілу номерів АВ в межах вибірки ----------------
def randomAM(n, iter):
    SAV = np.zeros((nAV))
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(0, iter)  # параметри закону задаются межами аргументу
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    # -------------- генерація номерів АВ за рівномірним законом  -------------------
    for i in range(nAV):
        SAV[i] = mt.ceil(
            np.random.randint(1, iter)
        )  # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
    print("номери АВ: SAV=", SAV)
    print("----- статистичны характеристики РІВНОМІРНОГО закону розподілу ВВ -----")
    print("матиматичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    print("-----------------------------------------------------------------------")
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return SAV


# ------------------------- нормальний закон розводілу ВВ ----------------------------
def randoNORM(dm, dsig, iter):
    S = np.random.normal(
        dm, dsig, iter
    )  # нормальний закон розподілу ВВ з вибіркою єбємом iter та параметрами: dm, dsig
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print("------- статистичны характеристики НОРМАЛЬНОЇ похибки вимірів -----")
    print("матиматичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    print("------------------------------------------------------------------")
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S


# ------------------- модель ідеального тренду (квадратичний закон)  ------------------
def Model(n):
    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = 0.0000005 * i * i  # квадратична модель реального процесу
    return S0


# ---------------- модель виміру (квадратичний закон) з нормальний шумом ---------------
def Model_NORM(SN, S0N, n):
    SV = np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV


# ----- модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
def Model_NORM_AV(S0, SV, nAV, Q_AV):
    SV_AV = SV
    SSAV = np.random.normal(
        dm, (Q_AV * dsig), nAV
    )  # аномальна випадкова похибка з нормальним законом
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = (
            S0[k] + SSAV[i]
        )  # аномальні вимірів з рівномірно розподіленими номерами
    return SV_AV


# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------
def Plot_AV(S0_L, SV_L, Text):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return


# ------------------------------ МНК згладжування -------------------------------------
def MNK(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


# ------------------------------ МНК згладжування -------------------------------------
def MNK_AV_Detect(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return C[1, 0]


# --------------------------- функція МНК ПРОГНОЗУВАННЯ -------------------------------
def MNK_Extrapol(S0, koef):
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    for i in range(iter + koef):
        Yout_Extrapol[i, 0] = (
            C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i)
        )  # проліноміальна крива МНК - прогнозування
    return Yout_Extrapol


# ------------------------------ МНК експонента -------------------------------------
def MNK_exponent(S0):
    iter = len(S0)
    Yout = np.zeros((iter, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 4))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    c0 = C[0, 0]
    c1 = C[1, 0]
    c2 = C[2, 0]
    c3 = C[3, 0]
    a3 = 3 * (c3 / c2)
    a2 = (2 * c2) / (a3**2)
    a0 = c0 - a2
    a1 = c1 - (a2 * a3)
    # iter = len(Yin)
    for i in range(iter):
        Yout[i, 0] = a0 + a1 * i + a2 * mt.exp(a3 * i)
    return Yout


# -------------------------------------------------- Expo_scipy -----------------------------------------------
def Expo_Regres(Yin, bstart):
    def func_exp(x, a, b, c, d):
        return a * np.exp(b * x) + c + (d * x)

    # ------ эмпирические коэффициенты старта для bstart=1202.059798705
    aStart = bstart / 10
    bStart = bstart / 1000
    cStart = bstart + 10
    dStart = bstart / 10
    iter = len(Yin)
    x_data = np.ones((iter))
    y_data = np.ones((iter))
    for i in range(iter):
        x_data[i] = i
        y_data[i] = Yin[i]
    # popt, pcov = curve_fit(func_exp, x_data, y_data, p0=(12, 0.0012, 1200, 120))
    popt, pcov = curve_fit(
        func_exp, x_data, y_data, p0=(aStart, bStart, cStart, dStart)
    )
    return func_exp(x_data, *popt)


# --------------------- функція обчислень алгоритму -а-в фільтру ------------------------
def ABF(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0 = 1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
    # -------------- початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra = Yin[0, 0] + Yspeed_retro
    alfa = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = (6 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])
    # -------------- рекурентний прохід по вимірам
    for i in range(1, iter):
        YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i, 0] + Yspeed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
    return YoutAB


# ------------------------------ Виявлення АВ за алгоритмом medium -------------------------------------
def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- еталон  ---------
    j = 0
    for i in range(n_Wind):
        l = j + i
        S0_Wind[i] = S0[l]
        dS_standart = np.var(S0_Wind)
        scvS_standart = mt.sqrt(dS_standart)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- детекція та заміна АВ --
        if scvS > (Q * scvS_standart):
            # детектор виявлення АВ
            # print('S0[l] !!!=', S0[l])
            S0[l] = mS
        # print('----- Вікно -----')
        # print('mS=', mS)
        # print('scvS=',scvS)
        # print('-----------------')
    return S0


# ------------------------------ Виявлення АВ за МНК -------------------------------------
def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- еталон  ---------
    Speed_standart = MNK_AV_Detect(SV_AV)
    Yout_S0 = MNK(SV_AV)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        #     Speed=MNK_AV_Detect(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- детекція та заміна АВ --
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
        # Speed_1 = abs(Speed / (Q*scvS))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        # print('Speed_standart=', Speed_standart_1)
        # print('Speed_1=', Speed_1)
        if Speed_1 > Speed_standart_1:
            # детектор виявлення АВ
            # print('S0[l] !!!=', S0[l])
            S0[l] = Yout_S0[l, 0]
    return S0


# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        Midi[l] = np.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi


# -------------------------------- БЛОК ОСНОВНИХ ВИКЛИКІВ ------------------------------
# ------------------------------ сегмент API (вхідних даних) ---------------------------
n = 10000
iter = int(n)  # кількість реалізацій ВВ
Q_AV = 3  # коефіцієнт переваги АВ
nAVv = 10
nAV = int((iter * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
dm = 0
dsig = 5  # параметри нормального закону розподілу ВВ: середне та СКВ
# ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
S0 = Model(n)  # модель ідеального тренду (квадратичний закон)
SAV = randomAM(n, iter)  # модель рівномірних номерів АВ
S = randoNORM(dm, dsig, iter)  # модель нормальних помилок
# ----------------------------- Нормальні похибки ------------------------------------
SV = Model_NORM(S, S0, n)  # модель тренда + нормальних помилок
Plot_AV(S0, SV, "квадратична модель + Норм. шум")
Stat_characteristics(SV, "Вибірка + Норм. шум")
# ----------------------------- Аномальні похибки ------------------------------------
SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV)  # модель тренда + нормальних помилок + АВ
Plot_AV(S0, SV_AV, "квадратична модель + Норм. шум + АВ")
Stat_characteristics(SV_AV, "Вибірка з АВ")
# ----------------- Очищення від аномальних похибок ковзним вікном --------------------
print("Оберіть метод виявлення та очищення вибірки від АВ та метод згладжування:")
print("1 - метод medium")
print("2 - метод MNK")
print("3 - метод sliding window")
print("4 - AB фільтр")
print("5 - МНК згладжування")
print("6 - МНК ПРОГНОЗУВАННЯ")
print("7 - МНК ЕКСПОНЕНТА")
print("8 - Регресія ЕКСПОНЕНТА")
mode = int(input("mode:"))

if mode == 1:
    print("Вибірка очищена від АВ метод medium")
    # --------- Увага!!! якість результату залежить від якості еталонного вікна -----------
    N_Wind_Av = 5  # розмір ковзного вікна для виявлення АВ
    Q = 1.6  # коефіцієнт виявлення АВ
    S_AV_Detect_medium = Sliding_Window_AV_Detect_medium(SV_AV, N_Wind_Av, Q)
    Plot_AV(S0, S_AV_Detect_medium, "Вибірка очищена від АВ алгоритм medium")
    Stat_characteristics(S_AV_Detect_medium, "Вибірка очищена від алгоритм medium АВ")
    Yout_SV_AV_Detect = MNK(S_AV_Detect_medium)
    Stat_characteristics(
        Yout_SV_AV_Detect, "МНК Вибірка відчищена від АВ алгоритм medium"
    )

if mode == 2:
    print("Вибірка очищена від АВ метод MNK")
    # ------------------- Очищення від аномальних похибок МНК --------------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    Q_MNK = 7  # коефіцієнт виявлення АВ
    S_AV_Detect_MNK = Sliding_Window_AV_Detect_MNK(SV_AV, Q_MNK, n_Wind)
    Plot_AV(S0, S_AV_Detect_MNK, "Вибірка очищена від АВ алгоритм MNK")
    Stat_characteristics(S_AV_Detect_MNK, "Вибірка очищена від АВ алгоритм MNK")
    Yout_SV_AV_Detect_MNK = MNK(S_AV_Detect_MNK)
    Stat_characteristics(
        Yout_SV_AV_Detect_MNK, "МНК Вибірка очищена від АВ алгоритм MNK"
    )

if mode == 3:
    print("Вибірка очищена від АВ метод sliding_wind")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = MNK(S_AV_Detect_sliding_wind)
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "МНК Вибірка очищена від АВ алгоритм sliding_wind",
    )

if mode == 4:
    print("ABF згладжена вибірка очищена від АВ алгоритм sliding_wind")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = ABF(S_AV_Detect_sliding_wind)
    Plot_AV(
        Yout_SV_AV_Detect_sliding_wind,
        S_AV_Detect_sliding_wind,
        "ABF Вибірка очищена від АВ алгоритм sliding_wind",
    )
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "ABF згладжена, вибірка очищена від АВ алгоритм sliding_wind",
    )

if mode == 5:
    print("MNK згладжена вибірка очищена від АВ алгоритм sliding_wind")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = MNK(S_AV_Detect_sliding_wind)
    Plot_AV(
        Yout_SV_AV_Detect_sliding_wind,
        S_AV_Detect_sliding_wind,
        "MNK Вибірка очищена від АВ алгоритм sliding_wind",
    )
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "MNK згладжена, вибірка очищена від АВ алгоритм sliding_wind",
    )

if mode == 6:
    print("MNK ПРОГНОЗУВАННЯ")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
    koef = mt.ceil(
        n * koef_Extrapol
    )  # інтервал прогнозу по кількісті вимірів статистичної вибірки
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = MNK_Extrapol(S_AV_Detect_sliding_wind, koef)
    Plot_AV(
        Yout_SV_AV_Detect_sliding_wind,
        S_AV_Detect_sliding_wind,
        "MNK ПРОГНОЗУВАННЯ: Вибірка очищена від АВ алгоритм sliding_wind",
    )
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "MNK ПРОГНОЗУВАННЯ, вибірка очищена від АВ алгоритм sliding_wind",
    )

if mode == 7:
    print("MNK ЕКСПОНЕНТА")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
    koef = mt.ceil(
        n * koef_Extrapol
    )  # інтервал прогнозу по кількісті вимірів статистичної вибірки
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = MNK_exponent(S_AV_Detect_sliding_wind)
    Plot_AV(
        Yout_SV_AV_Detect_sliding_wind,
        S_AV_Detect_sliding_wind,
        "MNK ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind",
    )
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "MNK ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind",
    )

if mode == 8:
    print("Регресія ЕКСПОНЕНТА")
    # --------------- Очищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
    koef = mt.ceil(
        n * koef_Extrapol
    )  # інтервал прогнозу по кількісті вимірів статистичної вибірки
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    Plot_AV(
        S0, S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Stat_characteristics(
        S_AV_Detect_sliding_wind, "Вибірка очищена від АВ алгоритм sliding_wind"
    )
    Yout_SV_AV_Detect_sliding_wind = Expo_Regres(S_AV_Detect_sliding_wind, 10)
    Plot_AV(
        Yout_SV_AV_Detect_sliding_wind,
        S_AV_Detect_sliding_wind,
        "Регресія ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind",
    )
    Stat_characteristics(
        Yout_SV_AV_Detect_sliding_wind,
        "Регресія ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind",
    )
