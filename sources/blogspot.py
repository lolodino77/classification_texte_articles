from blog import *


class Blogspot(Blog):

    def __init__(self, url, num_articles):
        Blog.__init__(self, url, num_articles)
        

    def __str__(self):