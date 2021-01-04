
import time
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from selenium.webdriver.chrome.options import Options
import os
from urllib.request import urlopen, Request
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
        self.conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='0000',
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
        IMPLICITLY_PAUSE_TIME = 120
        w=0.1
        self.driver.get(page_url)
        time.sleep(SCROLL_PAUSE_TIME)

        conn = self.conn
        curs = conn.cursor()


        # Get scroll height
#        last_height = driver.execute_script("return document.body.scrollHeight")
        last_height = 0
        count = 0
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        self.driver.execute_script("window.scrollTo( document.body.scrollHeight, document.body.scrollHeight-2000);")
        time.sleep(SCROLL_PAUSE_TIME)
        time.sleep(200)

        #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[1516]/div/div/div[2]/a/img
        while count <= 200:
            print(count)
            b = w*count
            count= count+1
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[2505]/div/div/div[2]/a/img

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)
            self.driver.execute_script("window.scrollTo( document.body.scrollHeight, document.body.scrollHeight-2000);")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)
            if new_height == last_height:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight-2000);")
                self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
                time.sleep(SCROLL_PAUSE_TIME+b)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
                time.sleep(SCROLL_PAUSE_TIME+b)

                new_height = self.driver.execute_script("return document.body.scrollHeight")
                self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
                time.sleep(SCROLL_PAUSE_TIME+b)
                if new_height == last_height:
                    break
            last_height = new_height

#//*[@id="react-view"]/div[3]/div[1]/div[2]/div[1]/div/div[4158]
#//*[@id="react-view"]/div[3]/div[1]/div[2]/div[1]/div/div[6282]/div/div/div[2]/a/img
        for i in range(1, nUrl+1):
            if i<200:
                continue
            try:
                xpath1 = '//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div/div[2]/a'
                url = self.driver.find_element_by_xpath(xpath1)
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
            print('cleanser error')
            text = 'None'
        return text



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

        projectHost = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[3]/div/div/div[1]/div/div/a').text

        SCROLL_PAUSE_TIME = 4
        IMPLICITLY_PAUSE_TIME = 120
        w=0.1
        # Get scroll height
        #        last_height = driver.execute_script("return document.body.scrollHeight")
        last_height = 0
        count = 0
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        self.driver.execute_script("window.scrollTo( document.body.scrollHeight, document.body.scrollHeight-2000);")
        time.sleep(SCROLL_PAUSE_TIME)
        #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[1516]/div/div/div[2]/a/img
        while True:
            print(count)
            b = w*count
            count= count+1
            #//*[@id="react-view"]/div[3]/div[1]/div[2]/div/div[2]/div[2505]/div/div/div[2]/a/img
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)
            self.driver.execute_script("window.scrollTo( document.body.scrollHeight, document.body.scrollHeight-2000);")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            self.driver.implicitly_wait(IMPLICITLY_PAUSE_TIME)
            time.sleep(SCROLL_PAUSE_TIME+b)
            if new_height == last_height:
                break
            last_height = new_height

        try:
            investment = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[2]/div/div[2]/div/div[3]/div/div/div[2]/div/div/div[1]').text
        except:
            investment = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[2]/div/div[2]/div/div[3]/div/div/section[1]/div/div[2]/div/div/div[1]').text
        investment = self.cleansing(investment)
        investment = re.sub('[^0-9]','',investment)
        for i in range(1, int(communityPostNum)):
            try:
                user = self.driver.find_element_by_xpath('//*[@id="react-view"]/div[5]/div[1]/div/div[1]/div/div/div/div['+str(i)+']/div/div[1]/div/div/div[1]/div/a/div').text
                print(user,end='')
                print(investment)
                if user != projectHost :
                    sql = "insert into tumblbug_user_info(site, title, username, investment) values ('tumblbug',\'%s\',\'%s\',\'%s\')"%(title, user, investment)
                    curs.execute(sql)

            except:
                continue
#//*[@id="react-view"]/div[5]/div[1]/div/div[1]/div/div/div/div[2]/div/div[1]/div[2]/div/div[1]/div/a/div
#//*[@id="react-view"]/div[5]/div[1]/div/div[1]/div/div/div/div[3]/div/div[1]/div/div/div[1]/div/a/div
#

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
            url = row[1]

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
            supporter = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-of-type(3) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA')
            remain_day = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-of-type(2) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA')
            percentachieved = sub_soup.select('#react-view > div.ProjectIntroduction__ProjectIntroductionBackground-sc-1o2ojgb-0.gsZkXT > div > div > aside > div.ProjectIntroduction__FundingStatus-sc-1o2ojgb-13.jqTlEc > div:nth-of-type(1) > div.ProjectIntroduction__StatusValue-sc-1o2ojgb-16.lgJcVA > span.ProjectIntroduction__FundingRate-sc-1o2ojgb-17.cNDicH')
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
            if split1[1] == split2:
                split2 = split1[1].split('원을')
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

            sql1 = 'insert into tumblbug_crawl (id, pagename, category, title, brand, achieve, funding, supporter, goal,remaining, endate)\
                                                    values(%d,\'%s\',\'%s\',\'%s\',\'%s\', \'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\')'\
                                                        %(id, pagename, category, title, Brand, achieve, funding, supporter, goalmoney, remaining, endate)
            curs.execute(sql1)
            conn.commit()

            print("title : ",end="")
            print(title)

            print("get user info func")
            self.getuserinfo(url, title)
            print('Crawling '+url+' finish',sql1)
            sql_url = "update tumblbug_urllist set status='펀딩중', crawled='T' where url=\'%s\'"% (url)  #where id -> where url
            curs.execute(sql_url)
            conn.commit()
            print("success3\n")

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

#커뮤니티1, SKKU
##react-view > div.ProjectPage__ProjectContentsBackground-f3cisk-0.lbhpFL > div.Container-gci8y7-0.wYlFP > div > div.ProjectPage__ProjectContentsMainColumn-f3cisk-2.dypJVt > div > div > div.Community__Posts-sc-1o9947p-0.kEeUNF > div.CommunityPostSummaryCard__PostSummaryCardWrapper-sc-3xh5y6-1.cHWozv > div > div.CommunityPostSummaryCard__MetaWrapper-sc-3xh5y6-2.Jpeyp > div > div > div.CommunityPostSummaryCard__UserProfile-sc-3xh5y6-6.ljvGXI > div > a > div #후원자1.

#커뮤니티8명, 암흑낭만주의
##react-view > div.ProjectPage__ProjectContentsBackground-f3cisk-0.lbhpFL > div.Container-gci8y7-0.wYlFP > div > div.ProjectPage__ProjectContentsMainColumn-f3cisk-2.dypJVt > div > div > div.Community__Posts-sc-1o9947p-0.kEeUNF > div:nth-of-type(2) > div > div.CommunityPostSummaryCard__MetaWrapper-sc-3xh5y6-2.Jpeyp > div > div > div.CommunityPostSummaryCard__UserProfile-sc-3xh5y6-6.ljvGXI > div > a > div #후원자1.
##react-view > div.ProjectPage__ProjectContentsBackground-f3cisk-0.lbhpFL > div.Container-gci8y7-0.wYlFP > div > div.ProjectPage__ProjectContentsMainColumn-f3cisk-2.dypJVt > div > div > div.Community__Posts-sc-1o9947p-0.kEeUNF > div:nth-of-type(3) > div > div.CommunityPostSummaryCard__MetaWrapper-sc-3xh5y6-2.Jpeyp > div.CommunityPostSummaryCard__Meta-sc-3xh5y6-4.ebxTKa > div > div.CommunityPostSummaryCard__UserProfile-sc-3xh5y6-6.ljvGXI > div > a > div #창작자.
##react-view > div.ProjectPage__ProjectContentsBackground-f3cisk-0.lbhpFL > div.Container-gci8y7-0.wYlFP > div > div.ProjectPage__ProjectContentsMainColumn-f3cisk-2.dypJVt > div > div > div.Community__Posts-sc-1o9947p-0.kEeUNF > div:nth-of-type(4) > div > div.CommunityPostSummaryCard__MetaWrapper-sc-3xh5y6-2.Jpeyp > div > div > div.CommunityPostSummaryCard__UserProfile-sc-3xh5y6-6.ljvGXI > div > a > div #후원자2.



if __name__ == '__main__':

    page_url = 'https://tumblbug.com/discover?achieveRate=1&currentMoney=1&sort=publishedAt'
    nUrl = 300
    wc = TumblbugCrawler()
    #wc.getUrlLister(page_url, nUrl)
    wc.tumblbugCrawler()


# In[ ]:
