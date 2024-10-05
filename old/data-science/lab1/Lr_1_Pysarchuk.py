# --------------------------- Lab_work_1  ------------------------------------

"""
Виконав: Олексій Писарчук
Lab_work_1, варіант 1, І рівень складності:
Закон зміни похибки – нормальний;
Закон зміни досліджуваного процесу – квадратичний.
Реальні дані – 1 показник на вибір.

Package                      Version
---------------------------- -----------

pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
xlrd                         2.0.1
matplotlib                   3.6.2
"""

import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt


# ------------------------ ФУНКЦІЯ парсингу реальних даних --------------------------


def file_parsing(URL, File_name, Data_name):
    """
    :param URL: адреса сайту для парсингу str
    :param File_name:
    :param Data_name:
    :return:
    """

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        # for name, values in d[[Data_name]].iteritems(): # приклад оновлення версій pandas для директиви iteritems
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print("Джерело даних: ", URL)
    return S_real


# ---------------------- ФУНКЦІЇ тестової адитивної моделі -------------------------


# ----------- рівномірний закон розподілу номерів АВ в межах вибірки ----------------
def randomAM(n):
    """

    :param n: кількість реалізацій ВВ - об'єм вибірки
    :return: номери АВ
    """

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
    print("----- статистичні характеристики РІВНОМІРНОГО закону розподілу ВВ -----")
    print("математичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    print("-----------------------------------------------------------------------")
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return SAV


# ------------------------- нормальний закон розподілу ВВ ----------------------------
def randoNORM(dm, dsig, iter):
    """
    :param dm:
    :param dsig:
    :param iter:
    :return:
    """

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
    """
    :param n:
    :return:
    """

    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = 0.0000005 * i * i  # квадратична модель реального процесу
        # S0[i] = 45  # квадратична модель реального процесу
    return S0


# ---------------- модель виміру (квадратичний закон) з нормальний шумом ---------------
def Model_NORM(SN, S0N, n):
    """
    :param SN:
    :param S0N:
    :param n:
    :return:
    """

    SV = np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV


# ----- модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
def Model_NORM_AV(S0, SV, nAV, Q_AV):
    """
    :param S0:
    :param SV:
    :param nAV:
    :param Q_AV:
    :return:
    """

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


def Stat_characteristics(SL, Text):
    """
    :param SL:
    :param Text:
    :return:
    """

    # статистичні характеристики вибірки з урахуванням тренду за МНК
    def Trend_MLS(SL):
        iter = len(SL)
        Yout = MNK_Stat_characteristics(SL)  # визначається за МНК
        SL0 = np.zeros((iter))
        for i in range(iter):
            SL0[i] = SL[i] - Yout[i, 0]
        return SL0

    # статистичні характеристики вибірки з урахуванням тренду за вихідними даними
    def Trend_Constant(SL):
        iter = len(SL)
        Yout = Model(iter)
        SL0 = np.zeros((iter))
        for i in range(iter):
            SL0[i] = SL[i] - Yout[i]
        return SL0

    SL0 = Trend_MLS(
        SL
    )  # статистичні характеристики вибірки з урахуванням тренду за МНК

    # SL0 = Trend_Constant(SL)    # статистичні характеристики вибірки з урахуванням тренду за вихідними даними

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print("------------", Text, "-------------")
    print("математичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    print("-----------------------------------------------------")
    return


# ------------- МНК згладжування визначення стат. характеристик -------------
def MNK_Stat_characteristics(S0):
    """
    :param S0:
    :return:
    """

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


# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------
def Plot_AV(S0_L, SV_L, Text):
    """
    :param S0_L:
    :param SV_L:
    :param Text:
    :return:
    """

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return


# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ----------------------------------

if __name__ == "__main__":

    # ------------------------------ сегмент констант ---------------------------------------
    n = 10000  # кількість реалізацій ВВ - об'єм вибірки
    iter = int(n)
    Q_AV = 3  # коефіцієнт переваги АВ
    nAVv = 10
    nAV = int((iter * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
    dm = 0
    dsig = 5  # параметри нормального закону розподілу ВВ: середнє та СКВ

    # ------------------------------ сегмент даних -------------------------------------------
    # ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
    S0 = Model(n)  # модель ідеального тренду (квадратичний закон)
    SAV = randomAM(n)  # модель рівномірних номерів АВ
    S = randoNORM(dm, dsig, iter)  # модель нормальних помилок

    # ----------------------------- Нормальні похибки -----------------------------------------
    SV = Model_NORM(S, S0, n)  # модель тренда + нормальних помилок
    Plot_AV(S0, SV, "квадратична модель + Норм. шум")
    Stat_characteristics(SV, "Вибірка + Норм. шум")

    # ----------------------------- Аномальні похибки -----------------------------------------
    SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV)  # модель тренда + нормальних помилок + АВ
    Plot_AV(S0, SV_AV, "квадратична модель + Норм. шум + АВ")
    Stat_characteristics(SV_AV, "Вибірка з АВ")

    # -------------------------------- Реальні дані -------------------------------------------
    # SV_AV = file_parsing(
    #     "https://www.oschadbank.ua/rates-archive",
    #     "data\\Oschadbank (USD).xls",
    #     "Купівля",
    # )
    SV_AV = file_parsing(
        "https://www.oschadbank.ua/rates-archive",
        "data\\Oschadbank (USD).xls",
        "Продаж",
    )
    # SV_AV = file_parsing(
    #     "https://www.oschadbank.ua/rates-archive",
    #     "data\\Oschadbank (USD).xls",
    #     "КурсНбу",
    # )
    Plot_AV(SV_AV, SV_AV, "Коливання курсу USD в 2022 році за даними Ощадбанк")
    Stat_characteristics(SV_AV, "Коливання курсу USD в 2022 році за даними Ощадбанк")


"""
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Задані характеристики вхідної вибірка:
часова надмірність даних із квадратичним законом;
статистичні характеристики:
    закон розподілу ВВ - нормальний
    n = 10000   # кількість реалізацій ВВ - об'єм вибірки   
    dm = 0
    dsig = 5    # параметри нормального закону розподілу ВВ: середнє та СКВ

2. Визначені характеристики вхідної вибірки:
часова надмірність даних із квадратичним законом підтверджена графіком;
статистичні характеристики:
    закон розподілу ВВ - нормальний, підтверджено гістограмою;
    -----------------------------------------------------------------------
    ------- статистичні характеристики НОРМАЛЬНОЇ похибки вимірів -----
    математичне сподівання ВВ= 0.023542519482130528
    дисперсія ВВ = 24.9006856891693
    СКВ ВВ= 4.9900586859444145
    ------------------------------------------------------------------
    ------------ Вибірка + Норм. шум -------------
    математичне сподівання ВВ= 0.003539378848703034
    дисперсія ВВ = 24.891878774923693
    СКВ ВВ= 4.989176161945346
    -----------------------------------------------------
    
3. Висновок
Відповідність заданих та обрахованих числових характеристик статистичної вибірки доводять адекватність розрахунків.
Розроблений скрипт можна використовувати для визначення статистичних характеристик реальних даних.
"""
