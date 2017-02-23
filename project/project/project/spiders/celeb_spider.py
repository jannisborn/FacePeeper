# coding=utf-8
import scrapy
from project.items import Celebrity
from scrapy_splash import SplashRequest
from scrapy_splash import SlotPolicy


class CelebritySpider(scrapy.Spider):
    name = "celebs"

    start_urls = ['http://www.imdb.com/search/name?gender=male,female']

    def parse(self, response):
        for celeb_page_url in response.css('div#main table.results tr.detailed td.image a::attr(href)').extract():
            yield scrapy.Request(response.urljoin(celeb_page_url), callback=self.parse_celeb_page)

    def parse_celeb_page(self, response):
        gallery_url = response.css('div#name-overview-widget div.see-more a::attr(href)').extract_first()
        yield scrapy.Request(response.urljoin(gallery_url), callback=self.parse_gallery)

    def parse_gallery(self, response):
        for img_thumbnail_url in response.css('div#media_index_thumbnail_grid a::attr(href)').extract():
            yield SplashRequest(response.urljoin(img_thumbnail_url), callback=self.parse_full_image,
                                args={
                                    'wait': 0.5,  # optional; parameters passed to Splash HTTP API
                                })

        next_page = response.css('div#right a.prevnext::attr(href)').extract()[-1]
        yield scrapy.Request(response.urljoin(next_page), callback=self.parse_gallery)

    def parse_full_image(self, response):
        img_url = response.css('img::attr(src)').extract_first()
        img_url = response.urljoin(img_url)
        name = response.css('span.mediaviewer_title::text').extract_first()
        yield Celebrity(name=name, image_urls=[img_url])

