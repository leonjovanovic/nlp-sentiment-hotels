from traceback import print_tb
import pandas as pd
import scrapy
from scrapy import signals
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.WARNING)
logging.getLogger('scrapy').setLevel(logging.WARNING)

class BookingReviewsSpider(scrapy.Spider):
    name = 'spider_booking_reviews'
    #start_urls = urls

    def __init__(self):
        self.driver = webdriver.Firefox()
        self.data = pd.DataFrame()
        self.start_hotel = 5000
        self.end_hotel = 6000

    def start_requests(self):
        urls = pd.read_json(f'hotels_data_all_cities.json', orient='index')['link'].to_list()[self.start_hotel:self.end_hotel]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'og_url': url,})

    def parse(self,response, og_url):

        delay = 5 # seconds
        #print(og_url)
        self.driver.get(response.url + '#tab-reviews')
        WebDriverWait(self.driver, delay).until(EC.url_to_be(response.url + '#tab-reviews'))
        xpath_positive = '//ul[@class="review_list"]//li[@class="review_list_new_item_block"]//div[@class="c-review-block"]//div[@class="bui-grid"]//div[@class="bui-grid__column-9 c-review-block__right"]//div[@class="c-review-block__row"]//div[@class="c-review"]//div[@class="c-review__row"]//p//span[@class="c-review__body"]'
        xpath_negative = '//ul[@class="review_list"]//li[@class="review_list_new_item_block"]//div[@class="c-review-block"]//div[@class="bui-grid"]//div[@class="bui-grid__column-9 c-review-block__right"]//div[@class="c-review-block__row"]//div[@class="c-review"]//div[@class="c-review__row lalala"]//p//span[@class="c-review__body"]'
        hotel_reviews = []
        try:
            WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, xpath_positive)))#'//div[@class="c-review"]//div[@class="c-review__row"]//p//span[@class="c-review__body"]'))).get_attribute('textContent')
            for pos_review in self.driver.find_elements_by_xpath(xpath_positive):
                if pos_review.get_attribute('textContent') != "Ovaj komentar nema tekst, samo ocenu":
                    hotel_reviews.append(pos_review.get_attribute('textContent').strip())
            for neg_review in self.driver.find_elements_by_xpath(xpath_negative):
                if neg_review.get_attribute('textContent') != "Ovaj komentar nema tekst, samo ocenu":
                    hotel_reviews.append(neg_review.get_attribute('textContent').strip())

            rows = pd.DataFrame([['hotels', og_url, review] for review in hotel_reviews], columns=['category', 'link', 'hotel_review'])
            self.data = pd.concat([self.data, rows], ignore_index=True)
            print(len(self.data))
        except TimeoutException:
            pass #print("Loading took too much time or didnt find element!")        
        try:
            WebDriverWait(self.driver, delay).until(EC.element_to_be_clickable((By.XPATH, '//a[@class="pagenext"]')))               
            next_page = self.driver.find_element_by_xpath('//a[@class="pagenext"]')
            if next_page:
                new_url = next_page.get_attribute('href')      
                yield response.follow(new_url, self.parse, cb_kwargs={'og_url': og_url,}) 
        except TimeoutException:
            pass #print(f"{len(self.data)}\n")


    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(BookingReviewsSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider


    def spider_closed(self, spider):
        print('Writing scraped DataFrame to JSON file')
        spider.logger.info('Writing scraped DataFrame to JSON file')
        out = self.data.to_json(orient='index', indent=4, force_ascii=False)
        with open(f'hotels_reviews_data_{self.start_hotel}_{self.end_hotel}.json', 'w', encoding='utf-8') as f:
            f.write(out)
        self.driver.close()
