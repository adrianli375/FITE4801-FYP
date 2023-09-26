import requests
from bs4 import BeautifulSoup
from csvlogging import Log

Logger = Log("wp.csv")

for i in range(1,101):
    result = requests.get(fr"https://wp2023.cs.hku.hk/fyp23{i:03d}/")
    if result.status_code != 200:
        continue
    soup = BeautifulSoup(result.text,"html.parser")
    if soup.find("main") is not None:
        headings = soup.find("main").getText().strip().replace("\n"," ")
        Logger.log(headings[:min(len(headings),150)],i)
    else:
        print("Error",i)
        Logger.log("Error",i)

print("done")