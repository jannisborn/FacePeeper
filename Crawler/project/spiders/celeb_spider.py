# coding=utf-8
import scrapy
from project.items import Celebrity
#from scrapy_splash import SplashRequest
#from scrapy_splash import SlotPolicy
import json


class CelebritySpider(scrapy.Spider):
    """A spider to crawl images from imdb.com
        
        
    """
    name = "celebs"

    wanted = [
        'Jürgen Vogel',
        'Benedict Cumberbatch',
        'Deepika Padukone',
        'Jason Statham',
        'Moritz Bleibtreu',
        'Til Schweiger',
        'Matthias Schweighöfer',
        'Daniel Brühl',
        'Megan Fox',
        'Margot Robbie',
        'Peter Stormare',
        'Jeffrey Dean Morgan',
        'Rowan Atkinson'
        ]  # 13
        
    def start_requests(self):
        start_url = 'http://www.imdb.com/search/name?gender=male,female&start={}'
    
        for i in range(1785):
            print('--------- Page {} of 2'.format(i))
            page_idx = (i * 50) + 1
            url = start_url.format(page_idx)
            yield scrapy.Request(url=url, callback=self.parse)
    

    def parse(self, response):
        for celeb_row in response.css('div#main table.results tr.detailed'):
            name = celeb_row.css('td.name a::text').extract_first()
            
            if name in self.wanted:
                meta = {'name': name}
                link = celeb_row.css('td.image a::attr(href)').extract_first()
                link = response.urljoin(link)
                
                request = scrapy.Request(response.urljoin(link), callback=self.parse_celeb_page, meta=meta)

                yield request


    def parse_celeb_page(self, response):
        gallery_url = response.css('div#name-overview-widget div.see-more a::attr(href)').extract_first()
        
        yield scrapy.Request(response.urljoin(gallery_url), callback=self.parse_gallery, meta=response.request.meta)


    def parse_gallery(self, response):
#        for img_thumbnail_url in response.css('div#media_index_thumbnail_grid a'):
        img_thumbnail_url = response.css('div#media_index_thumbnail_grid a::attr(href)').extract_first()

        yield scrapy.Request(response.urljoin(img_thumbnail_url), callback=self.parse_full_image, meta=response.request.meta)


#        next_page = response.css('div#right a.prevnext::attr(href)')
#        
#        if len(next_page) > 1:
#            next_page = next_page.extract()[-1]
#
#            yield scrapy.Request(response.urljoin(next_page), callback=self.parse_gallery, meta=response.request.meta)


    def parse_full_image(self, response):
        image_json = response.css('script#imageJson::text').extract_first()
        image_data = json.loads(image_json)
        idx = 0
        while True:
            try:
                print('trying')
                img_url = image_data['mediaViewerModel']['allImages'][idx]['src']
                
                print('******************************* parse image *****************************')
                print('name: {}'.format(response.request.meta['name']))
                print('url: {}'.format(img_url))

                
                yield Celebrity(name=response.request.meta['name'], img_urls=[img_url])
            except KeyError:
                break
            idx += 1

