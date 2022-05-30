import scrapy

class BookingSpider(scrapy.Spider):
    name = "spider_booking"

    def start_requests(self):
        urls = [
            'https://www.booking.com/searchresults.sr.html?city=-74897',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'})

    def parse(self, response):
        i = 0
        for selector in response.xpath('//div[@data-testid="property-card"]//div//div//div//a'):
            if 'aria-hidden' in selector.attrib.keys():
                print(i)
                href = selector.attrib['href']
                i += 1
                yield response.follow(href, self.parse_hotel, headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'})
        
        if '&offset=' not in response.url:
            pagination_offset = 25
            for i in range(1, 40):
                new_url = response.url + '&offset=' + str(pagination_offset * i)
                yield response.follow(new_url, self.parse, headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'})

    def parse_hotel(self, response):
        print(response.url.split("/")[-1][:15])