"""
Приклад парсингу табличних / числових даних з  сайту "Minfin.com.ua", що характеризують
величину прожиткового мінімуму в Україні за період 2000 - 2023 рр. (дані подані у грн.).
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs


def parsing_site_minfin(URL):
    """
    The function that parses data and save it into the file
    :param URL: class 'str'
    :return: class 'list'
    """

    r = requests.get(URL)
    print(r.status_code)
    soup = bs(r.text, "html.parser")
    tables = [
        [
            [td.get_text(strip=True) for td in tr.find_all("td")]
            for tr in table.find_all("tr")
        ]
        for table in soup.find_all("table")
    ]
    table_data = tables[0]
    return table_data


# ---------------------------- Функція отримання реальних даних із файлу -----------------------
def file_parsing(URL, File_name, Data_name):
    """
    The function of receiving real data from a file
    :param URL: class 'str'
    :param File_name: class 'str'
    :param Data_name: class 'str'
    :return: class 'np.ndarray'
    """

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print("Джерело даних: ", URL)
    return S_real


if __name__ == "__main__":
    # ------------------------ Парсинг даних з сайту Мінфін ----------------------------------

    URL = "https://index.minfin.com.ua/ua/labour/wagemin/"
    print(URL)
    print(parsing_site_minfin(URL))

    columns_names = [
        "Period",
        "Total",
        "Children < 6 years old",
        "Children 6 - 18 years old",
        "Persons capable of working",
        "Persons who have lost working capacity",
    ]

    df = pd.DataFrame(data=parsing_site_minfin(URL), columns=columns_names)
    print(df)

    df = df.dropna()
    df = df.iloc[::-1]

    df.to_excel("output\\Minfin_LivingWage.xlsx")
