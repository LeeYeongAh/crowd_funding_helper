import time
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from selenium.webdriver.chrome.options import Options
import os
from urllib.request import urlopen
from lxml import etree
import pymysql
from datetime import datetime
import re

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

class TumblbugCrawler:
    def __init__(self):
        self.conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='wdta2181',
                               db='test', charset='utf8')

        #self.path = os.path.dirname(os.path.realpath(__file__))
        print('DB connected')
        option = Options()

        option.add_argument("--disable-infobars")
        option.add_argument("start-maximized")
        option.add_argument("--disable-extensions")

        # Pass the argument 1 to allow and 2 to block
        option.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2
        })

        self.path = os.path.dirname(os.path.realpath(__file__))
        self.driver = webdriver.Chrome(options=option, executable_path=self.path + "\chromedriver.exe")


    def getUrlLister(self, page_url, nUrl):

        #chromedriver = 'C:/Users/ljy01/Desktop/chromedriver/chromedriver_win32/chromedriver.exe'
        #driver = webdriver.Chrome(chromedriver)
        SCROLL_PAUSE_TIME = 4

        self.driver.get(page_url)
        time.sleep(SCROLL_PAUSE_TIME)

        conn = self.conn
        curs = conn.cursor()

        #nUrl을 100개 고정하지 않고, xpath로 프로젝트 개수 가져옴
        projectNum = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[1]/span')
        projectNum = projectNum.text
        nUrl = int(projectNum)
        print('nUrl : ', nUrl)
        # Get scroll height
#        last_height = driver.execute_script("return document.body.scrollHeight")
        last_height = 0

        # Get scroll height
#        last_height = driver.execute_script("return document.body.scrollHeight")
        last_height = 0

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            #print('down')
            time.sleep(SCROLL_PAUSE_TIME)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight-600);")
            #print('up')
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        for i in range(1, nUrl+1):
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div/div[3]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[3]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[3]/div/div/div[3]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[4]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[5]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[6]/div/div/div[3]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[7]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[8]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[9]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[10]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[11]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[12]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[13]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[14]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[15]/div/div/div[2]/a
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[16]/div/div/div[2]/a
            try:
                xpath1 = '//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div/div[2]/a'
                url = self.driver.find_element_by_xpath(xpath1)
                url = url.get_attribute('href')
            except:
                xpath1 = '//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div/div[3]/a'
                url = self.driver.find_element_by_xpath(xpath1)
                url = url.get_attribute('href')
            sql0 = "select * from tumblbug_urllist where url=\'%s\'" % (url)
            curs.execute(sql0)
            rows = curs.fetchall()
            status = 'F'
            if len(rows)==0:
                sql= "insert into tumblbug_urllist(url, crawled) values (\'%s\',\'%s\')"%(url,status)
                curs.execute(sql)
                conn.commit()

                print(i,url)
                    #url = url.get_attribute('href')
                    #print(url)
    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #conn.close()
    """
    def goToCommunityTab(self):
        self.driver.find_element_by_xpath('//*[@id="contentsNavigation"]/nav/div/div/a[2]').click()
        self.driver.implicitly_wait(30)

        #html = self.driver.page_source
        #soup = BeautifulSoup(html, 'html.parser')
    """
    #https://tumblbug.com/abcfrancais/story?ref=discover    #스토리 탭 눌렀을때.
    #https://tumblbug.com/abcfrancais/community?ref=discover

    #https://tumblbug.com/ink?ref=discover  #기본 페이지.
    #https://tumblbug.com/ink/community?ref=discover
    def getuserinfo(self, url, title):
        conn = self.conn
        curs = conn.cursor()
        #https://tumblbug.com/planner101?ref=discover
        #https://tumblbug.com/planner101/community?ref=discover
        print("getuserinfo")
        url = url.replace("?","/community?")
        #req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        #response = urlopen(req)
        self.driver.get(url)
        #htmlparser = etree.HTMLParser()
        #tree = etree.parse(response, htmlparser)
        #projectHost = self.cleansing(projectHost)
        print("is there any posting?")
        try:
            communityPostNum = self.driver.find_element_by_xpath('//*[@id="contentsNavigation"]/nav/div/div/a[2]/span').text
            self.driver.implicitly_wait(5)
            print("yes, there are ", end='')
            print(communityPostNum)
        except:
            print("no")
            return

        projectHost = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[2]/div/div[1]/div/div[2]/a/span').text

        try:    #진행 중인 프로젝트.
            investment = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[2]/div/div[2]/div/div[3]/div/div/section[1]/div/div[2]/div/div/div[1]').text
        except : #종료된 프로젝트.
            investment = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[2]/div/div[2]/div/div[3]/div/div/div[2]/div/div/div[1]').text

        investment = self.cleansing(investment)
        investment = re.sub('[^0-9]','',investment)

        for i in range(2, int(communityPostNum)+2):
            try:
                user = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[1]/div/div/div/div['+str(i)+']/div/div[1]/div/div/div[1]/div/a/div').text
                print(user,end='')
                print(investment)
                if user != projectHost :
                    sql = "insert into tumblbug_user_info(site, title, username, investment) values ('tumblbug',\'%s\',\'%s\',\'%s\')"%(title, user, investment)
                    curs.execute(sql)

            except:
                continue

            """
            print(user,end='')
            print(investment)
            sql = "insert into tumblbug_user_info(site, title, username, investment) values ('tumblbug',\'%s\',\'%s\',\'%s\')"%(title, user, investment)

            curs.execute(sql)
            """
            conn.commit()



    def cleansing(self, text):
        try:
            text = re.sub('[ㅣ,#/:$@*\"※&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]', '', text)

            text = re.sub(r'\\', '', text)
            text = re.sub(r'\\\\', '', text)
            text = re.sub('\'', '', text)
            text = re.sub('\"', '', text)

            text = re.sub('\u200b', ' ', text)
            text = re.sub('&nbsp;|\t', ' ', text)
            text = re.sub('\r\n', '\n', text)

            while (True):
                text = re.sub('  ', ' ', text)
                if text.count('  ') == 0:
                    break

            while (True):
                text = re.sub('\n \n ', '\n', text)
                # print(text.count('\n \n '))
                if text.count('\n \n ') == 0:
                    break

            while (True):
                text = re.sub(' \n', '\n', text)
                if text.count(' \n') == 0:
                    break

            while (True):
                text = re.sub('\n ', '\n', text)
                if text.count('\n ') == 0:
                    break

            while (True):
                text = re.sub('\n\n', '\n', text)
                # print(text.count('\n\n'))
                if text.count('\n\n') == 0:
                    break
            text = re.sub(u'[\u2500-\u2BEF]', '', text)  # I changed this to exclude chinese char

            # dingbats
            text = re.sub('\\-|\]|\{|\}|\(|\)', "", text)

            text = re.sub(u'[\u2702-\u27b0]', '', text)
            text = re.sub(u'[\uD800-\uDFFF]', '', text)
            text = re.sub(u'[\U0001F600-\U0001F64F]', '', text)  # emoticons
            text = re.sub(u'[\U0001F300-\U0001F5FF]', '', text)  # symbols & pictographs
            text = re.sub(u'[\U0001F680-\U0001F6FF]', '', text)  # transport & map symbols
            text = re.sub(u'[\U0001F1E0-\U0001F1FF]', '', text)  # flags (iOS)
        except Exception as e:
            print('cleaser error')
            text = 'None'
        return text

    def tumblbugCrawler(self):
        conn = self.conn
        curs = conn.cursor()

        sql = "select * from tumblbug_urllist where crawled='F' or crawled='DB insert error'"
        curs.execute(sql)
        rows = curs.fetchall()

        #self.driver = self.webdriver.Chrome(options=option, executable_path=self.path + "\chromedriver.exe")
        #chromedriver = 'C:/Users/ljy01/Desktop/chromedriver/chromedriver_win32/chromedriver.exe'
        #driver = webdriver.Chrome(chromedriver)

        for row in rows:
            id = row[0]
            pagename = "tumblbug"#row[2]
            url = row[2]

            if url == 'None':
                continue;

            self.driver.get(url)
            sub_html = self.driver.page_source

            sub_soup = BeautifulSoup(sub_html, 'html.parser')
            sub_html = self.driver.page_source

            sub_soup = BeautifulSoup(sub_html, 'html.parser')

            projectName = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > div.ProjectIntroduction__ProjectOutline-sc-1o2ojgb-2.jbdzfG > div > h1')
            brand = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > div.ProjectIntroduction__ProjectOutline-sc-1o2ojgb-2.jbdzfG > div > div > a')
            Category = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > div.ProjectIntroduction__ProjectOutline-sc-1o2ojgb-2.jbdzfG > div > a > span')
            collection = sub_soup.find('div',attrs={'class': 'ProjectIntroduction__StatusValue-sc-1o2ojgb-16 lgJcVA'})
            supporter = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-child(3) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA')
            remain_day = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-child(2) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA')
            percentachieved = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-child(1) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA > span.ProjectIntroduction__FundingRate-sc-1o2ojgb-17.cNDicH')
            goal = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.FundingInformation-cjd67l-0.gGtZns > div > span')
            pagename = "tumblbug"

            print("success1\n")
            #category
            for i in Category:
                category = i.text.strip()
            category = category.replace(" ","")

            #title
            for i in projectName:
                title = i.text.strip()

            #brand
            for i in brand:
                global Brand
                Brand = i.text.strip()

            #achieve
            for i in percentachieved:
                pt = i.text.strip()
            ptnum = len(pt)
            achieve = pt[:ptnum-1]

            #funding - 달성 금액
            for tag in collection(['span']):
                tag.replace_with('')
            funding = collection.text.replace(",","")


            #supporter = "tobe"

            for i in supporter :
                supporternum = i.text.strip()
            supporter = supporternum[:-1]
            supporter = supporter.replace(',', '')

            # likes = NULL

            # goal
            for i in goal:
                notsplit = i.text.strip()
            split1 = notsplit.split("목표 금액인")
            split2 = split1[1].split('원이')
            goalmoney = split2[0].replace(",","").replace(" ","")

            #remaining
            for i in remain_day:
                day = i.text.strip()
            daynum = len(day)
            remaining = day[:daynum-1]
            if remaining[-1] == '시' :
                remaining = remaining[:-1]

            #endate - 수작업
            endate = "tobe"

            pagename = self.cleansing(pagename)
            category = self.cleansing(category)
            title= self.cleansing(title)
            Brand = self.cleansing(Brand)
            achieve= self.cleansing(achieve)
            funding= self.cleansing(funding)
            supporter= self.cleansing(supporter)
            goalmoney= self.cleansing(goalmoney)
            remaining= self.cleansing(remaining)
            endate= self.cleansing(endate)


            print("success2\n")

            sql1 = 'insert into tumblbug_crawl (id, pagename, category, title, brand, achieve, funding, supporter, goal,remaining_day, endate)\
                                                    values(%d,\'%s\',\'%s\',\'%s\',\'%s\', \'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\')'\
                                                        %(id, pagename, category, title, Brand, achieve, funding, supporter, goalmoney, remaining, endate)
            curs.execute(sql1)
            conn.commit()
            print("success3\n")
            print('Crawling '+url+' finish',sql1)

            conn.commit()
            sql_url = "update tumblbug_urllist set status='펀딩중', crawled='T' where url=\'%s\'"% (url)  #where id -> where url
            curs.execute(sql_url)
            conn.commit()

            #self.goToCommunityTab()
            self.getuserinfo(url, title)

        conn.close()

"""
        n_scrollDown = nUrl//18+2
        k=0
        while k<n_scrollDown+1:
            k+=1
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-100);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if k == n_scrollDown:
                break
            last_height = new_height
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-100);")
            print(str(k)+"번 반복했습니다.")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-100);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-100);")
"""


if __name__ == '__main__':

    page_url = 'https://tumblbug.com/discover?ongoing=onGoing&sort=endedAt'
    nUrl = 100
    wc = TumblbugCrawler()
    #wc.getUrlLister(page_url, nUrl)
    wc.tumblbugCrawler()
