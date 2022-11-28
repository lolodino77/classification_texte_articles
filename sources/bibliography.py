from article import *
from datasource import *


class Bibliography(DataSource):
    def __init__(self, url, topic, num_articles):
        DataSource.__init__(self, url, topic, num_articles)
        self.articles_urls = self.create_articles_urls()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant la bibliographie """
        print("str :")
        str_articles_urls = str(self.articles_urls)
        desc = DataSource.__str__(self)
        desc += "\narticles_urls = " + str_articles_urls
        return(desc)   


    def create_articles_urls(self):
        """ Recupere la liste des urls (liens hypertextes) presents sur une page internet.
        
        Parametres:
        url (string) : L'url de la page internet dont on veut recuperer les urls
        filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
        file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
        
        Sortie:
        urls (liste de string) : Une liste d'urls + (ecriture du resultat dans le fichier filename)
        """
        #Recupere le texte de la page web a l'aide d'un parser
        reqs = requests.get(self.url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        #Recupere un par un tous les liens url presents sur l'article
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