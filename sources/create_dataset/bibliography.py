from article import *
from datasource import *


class Bibliography(DataSource):
    def __init__(self, url, corpus_name, num_articles):
        """ Constructeur de la classe Bibliography """
        DataSource.__init__(self, url, num_articles)
        self.corpus_name = corpus_name
        self.filename_corpus_txt = self.create_corpus_txt_filename()
        self.path_corpus_txt = "./data/input/corpus_txt/" + self.filename_corpus_txt 
        self.path_articles_urls = "./data/input/articles_lists/articles_list_{}.txt".format(self.corpus_name)
        self.articles_urls = self.create_articles_urls()


    def __str__(self):
        """ Descripteur de la classe Bibliography """
        print("str :")
        str_corpus_name = str(self.corpus_name)
        str_path_articles_urls = str(self.path_articles_urls)
        str_articles_urls = str(self.articles_urls)
        str_path_corpus_txt = str(self.path_corpus_txt)
        desc = DataSource.__str__(self)
        desc += "\ncorpus_name = " + str_corpus_name
        desc += "\npath_corpus_txt = " + str_path_corpus_txt
        desc += "\npath_articles_urls = " + str_path_articles_urls
        desc += "\narticles_urls = " + str_articles_urls
        return(desc)


    def create_articles_urls(self):
        """ Recupere et donne la liste des urls (liens hypertextes) presents sur une page internet.
        
        Parametres:
            url (string) : L'url de la page internet dont on veut recuperer les urls
            filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
            file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
        
        Sortie:
            urls (liste de string) : Une liste d'urls + (ecriture du resultat dans le fichier filename)
        """
        # Recupere le texte de la page web a l'aide d'un parser
        reqs = requests.get(self.url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        # Recupere un par un tous les liens url presents sur l'article
        urls = []
        for link in soup.find_all('a'):
            link_str = str(link)
            # print("link =", str(link))
            # print("type(link) =", type(link))
            if("https" in link_str):
                url_i = link.get('href')
                if("/20" in url_i): # condition si c'est un article (20 = 2 premiers chiffres des annees 2021, 2010...)
                    urls.append(url_i)
        urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

        return(urls)