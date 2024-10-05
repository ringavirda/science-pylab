# ------------------  HTTP: для парсингу сайтів ----------------------------
"""
Приклад
парсингу сайтів із збереженням інформації до файлів різного формату
df.to_csv("output.csv")
df.to_excel("output.xlsx")
df.to_json("output.json")

"""

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd


def Parsing_Site_work_ua(URL_TEMPLATE):
    """
    site parsing python
    web scraping / site scraping python
    Data scraping - швидше очищення та підготовка даних
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html

    :param URL_TEMPLATE: URL Site work.ua
    :return: class 'dict'
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    """
    для методу get: headers=headers - МИ відрекомендувались звичайним браузером при зверненні до серверу.
    """
    r = requests.get(URL_TEMPLATE, headers=headers)
    print(r.status_code)
    print(r.text)
    soup = bs(r.text, "html.parser")
    vacancies = soup.find_all("div", class_="card")
    result_list = []
    for vacancy in vacancies:
        name = vacancy.find("h2")
        if name is None or name.a is None:
            continue
        title = name.a["title"]
        href = "https://www.work.ua" + name.a["href"]
        about = vacancy.p.text.strip()

        result_list.append({"title": title, "href": href, "about": about})

    print(result_list)
    print(type(result_list))

    return result_list


URL_TEMPLATE = "https://www.work.ua/jobs-data+scientist/?page=1"

df = pd.DataFrame(data=Parsing_Site_work_ua(URL_TEMPLATE))
print("df = ", df)
df.to_csv("output\\output.csv")
df.to_excel("output\\output.xlsx")
df.to_json("output\\output.json")
