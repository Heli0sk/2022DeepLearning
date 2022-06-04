import json
import requests
import time
import pandas as pd


def requestData(url, datatype, starttime, endtime):
    """
    :param url:
    :param datatype: pigprice, maizeprice, bean
    :param starttime: start time, for example: 2021-06-02
    :param endtime:
    :return:
    """
    header = {
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/80.0.3987.132 Safari/537.36'
    }
    req_url = url + datatype + "&sDate=" + starttime + "&eDate=" + endtime
    s = requests.session()
    s.keep_alive = False

    txt = requests.get(url=req_url, headers=header)
    js = json.loads(txt.text)

    orgData = js['data']
    print(orgData)
    pigPrice = []
    for item in orgData:
        pigPrice.append(item['price'])

    data = {
        datatype: pigPrice
    }
    data = pd.DataFrame(data)
    print(data.head())
    data.to_csv(datatype+".csv", index=False)


if __name__ == '__main__':
    url = "https://api.yangzhu360.com/zhujia/api/line?areaId=-1&type="
    requestData(url, "pigprice", "2019-06-02", "2022-06-02")
    # time.sleep(10)
    # requestData(url, "maizeprice", "2019-06-02", "2022-06-02")
    # time.sleep(10)
    # requestData(url, "bean", "2019-06-02", "2022-06-02")

