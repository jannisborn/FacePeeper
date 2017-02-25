# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import hashlib
import scrapy
from scrapy.exceptions import DropItem
from scrapy.utils.python import to_bytes
from scrapy.pipelines.images import ImagesPipeline


class ProjectPipeline(ImagesPipeline):


    def file_path(self, request, response=None, info=None):
        url = request.url
        image_guid = hashlib.sha1(to_bytes(url)).hexdigest()

        name = request.meta.get('name')

        return '{name}/{hash}.jpg'.format(name=name, hash=image_guid)


    def get_media_requests(self, item, info):
        return [scrapy.Request(url, meta={'name': item['name']})
                for url in item['img_urls']]
