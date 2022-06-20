from typing import Set
import scrapy
import json
import pandas as pd
from scrapy import signals

class BookingSpider(scrapy.Spider):
    name = "spider_booking"
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.data = pd.DataFrame()
        self.unique_links = set()
        self.number_of_crawled = 0

    def start_requests(self):
        # ----------------------- EDIT THIS PART -----------------------
        self.city = "all_cities"
        urls = [
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&ssne=Beograd&ssne_untouched=Beograd&lang=sr&src=searchresults&group_adults=2&no_rooms=1&group_children=0&sb_travel_purpose=leisure&nflt=uf%3D-85598%3Buf%3D-99989%3Buf%3D-99322%3Buf%3D-76924%3Buf%3D-84515%3Buf%3D-91364%3Buf%3D-81046%3Buf%3D-99272%3Buf%3D365941%3Buf%3D-96370%3Buf%3D-83220',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&ssne=Beograd&ssne_untouched=Beograd&lang=sr&src=searchresults&group_adults=2&no_rooms=1&group_children=0&sb_travel_purpose=leisure&nflt=uf%3D-74221%3Buf%3D-74377%3Buf%3D-90243%3Buf%3D-101070%3Buf%3D-73933%3Buf%3D-97489',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=uf%3D-90115%3Buf%3D-78471%3Buf%3D-96302',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=uf%3D-101185',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=uf%3D-90313',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D5%3Bclass%3D1%3Bclass%3D2%3Bclass%3D3%3Buf%3D-74897%3Bclass%3D0',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D4%3Buf%3D-74897',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D5%3Bclass%3D1%3Bclass%3D2%3Bclass%3D3%3Buf%3D362444%3Bclass%3D4',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D0%3Buf%3D362444',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=ht_id%3D220',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=popular_activities%3D70%3Bclass%3D3',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=popular_activities%3D70%3Bclass%3D0',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D2%3Bclass%3D1%3Bclass%3D5%3Bpopular_activities%3D70%3Bclass%3D4',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15705%3Btop_destinations%3D15703%3Btop_destinations%3D15702%3Btop_destinations%3D15699%3Btop_destinations%3D15698%3Btop_destinations%3D15697%3Btop_destinations%3D15696',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15692%3Btop_destinations%3D15694%3Btop_destinations%3D15700',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15701',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15704%3Bclass%3D0',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D2%3Bclass%3D4%3Bclass%3D3%3Bclass%3D5%3Btop_destinations%3D15704%3Bclass%3D1',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15695%3Bclass%3D0',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D4%3Bclass%3D3%3Bclass%3D5%3Btop_destinations%3D15695%3Bclass%3D2',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D5%3Bclass%3D0%3Btop_destinations%3D3252%3Bclass%3D4',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D1%3Bclass%3D3%3Btop_destinations%3D3252%3Bclass%3D2',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D15706%3Bclass%3D3',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D2%3Bclass%3D4%3Bclass%3D1%3Btop_destinations%3D15706%3Bclass%3D5',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D0%3Btop_destinations%3D15706%3Breview_score%3D90',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=class%3D0%3Btop_destinations%3D15706%3Breview_score%3D60',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D2',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D3',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D4',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D5',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=hotelfacility%3D8%3Btop_destinations%3D3399',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=hotelfacility%3D11%3Btop_destinations%3D3399',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D17',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D16',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D25',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D54',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D107',
            'https://www.booking.com/searchresults.sr.html?aid=304142&ss=Srbija&nflt=top_destinations%3D3399%3Bhotelfacility%3D301',
        ]
        # --------------------------------------------------------------
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for selector in response.xpath('//div[@data-testid="property-card"]//div//div//div//a'):
            if 'aria-hidden' in selector.attrib.keys():
                href = selector.attrib['href'].split('?')[0] if '?' in selector.attrib['href'] else selector.attrib['href']
                href_hotel = href.split('/')[5].split('html')[0]
                if href_hotel not in self.unique_links:
                    self.unique_links.add(href_hotel)
                    yield response.follow(href, self.parse_hotel)

        if '&offset=' not in response.url:            
            num_of_pages = int(response.xpath('//div[@data-testid="pagination"]//nav//div')[1].xpath('//ol//li//text()').getall()[-1])
            pagination_offset = 25
            for i in range(1, num_of_pages):
                new_url = response.url + '&offset=' + str(pagination_offset * i)
                yield response.follow(new_url, self.parse)

    def parse_hotel(self, response):
        hotel_name = response.xpath('//h2[@class="hp__hotel-name"]/text()').getall()[1].strip('\n')
        hotel_descriptions = response.xpath('//div[@id="property_description_content"]//p//text()').getall()
        if hotel_descriptions[0][:35] == 'Ispunjavate uslove za Genius popust':
            hotel_descriptions = hotel_descriptions[3:]
        row = pd.DataFrame([['hotels', response.url, hotel_name, ' '.join(hotel_descriptions)]], columns=['category', 'link', 'hotel_name', 'hotel_description'])
        self.data = pd.concat([self.data, row], ignore_index = True)
        self.number_of_crawled += 1
        print(f'Trenutan broj hotela je {self.number_of_crawled}')
        

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(BookingSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider


    def spider_closed(self, spider):
        spider.logger.info('Writing scraped DataFrame to JSON file')
        out = self.data.to_json(orient='index', indent=4, force_ascii=False)
        with open(f'hotels_data_{self.city}.json', 'w', encoding='utf-8') as f:
            f.write(out)
