from article import *


class Bibliography(Article):
    def __init__(self, url, topic, num_articles):
        Article.__init__(self, url, topic)
        self.articles_urls = self.create_articles_urls()
        self.num_articles = num_articles
        self.all_articles = self.create_all_articles()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant la bibliographie """
        print("str :")
        str_articles_urls = str(self.articles_urls)
        str_num_articles = str(self.num_articles)
        str_all_articles = str(self.all_articles)
        desc = Article.__str__(self)
        desc += "\narticles_urls = " + str_articles_urls
        desc += "\nnum_articles = " + str_num_articles
        desc += "\nall_articles = " + str_all_articles
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


    def create_all_articles(self):
        """ Recupere la variable all_articles depuis la commande de terminal """
        if(self.num_articles == "all"):
            all_articles = True
        else:
            all_articles = False

        return(all_articles)


    # def create_num_articles():
    #     """ Recupere la variable num_articles depuis la commande de terminal """
    #     # num_articles = sys_argv[1]
    #     if(self.num_articles != "all"):
    #         num_articles = int(num_articles) 

    #     return(num_articles)


    def save_paragraphs(self, savemode="overwrite"):
        """Cree un corpus d'un topic (au format de liste de documents/textes) dans le fichier texte filename_output
        a partir d'une liste d'adresses urls d'articles

        Parametres: 
        articles_urls (liste de string) : La liste d'urls d'articles dont on veut extraire les paragraphes. 
                                            Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
        path_articles_list (string) : La liste des paths des listes d'articles
        path_corpus (string) : Le path vers le corpus, exemple = 
        save_mode (string) : Le mode d'ecriture du fichier ("append" = ajouter ou "overwrite" = creer un nouveau)
        
        Sortie:
        None : Fichier filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne

        Done : version "overwrite" recreer le corpus a chaque fois de zero 
        To do : version "append" ajouter du texte a un corpus deja cree, version "ignore" ne fais rien si fichier existe deja
                version "error" qui renvoie une erreur si fichier existe deja
        """
        #Ecrit dans le fichier texte filename_corpus.txt tous les paragraphes tous les articles d'une liste
        if(not self.all_articles):
            self.articles_urls = self.articles_urls[:self.num_articles] # garder que les num_articles premiers articles
            # rajouter cas ou il n'y a qu'un seul article
        if(savemode == "overwrite"):
            # print("path_corpus =", path_corpus)
            # print("articles_urls =", articles_urls)
            article_url = self.articles_urls[0]
            article = Article(article_url, self.topic)
            print("article_url =")
            print(article_url)
            # paragraphs = get_paragraphs_of_article(article_url)
            article.save_paragraphs(file_open_mode="w", sep = "\n\n")
            # save_paragraphs(paragraphs, path_corpus, file_open_mode="w", sep = "\n\n")
            for article_url in self.articles_urls[1:]:
                article = Article(article_url, self.topic)
                article.save_paragraphs(file_open_mode="a", sep = "\n\n")

                print("article_url =")
                print(article_url)
                # paragraphs = get_paragraphs_of_article(article_url)
                # save_paragraphs(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n")
        elif(savemode == "append"):
            for article_url in self.articles_urls:
                print("article_url =")
                print(article_url)
                article = Article(article_url, self.topic)
                article.save_paragraphs(file_open_mode="a", sep = "\n\n")

                # paragraphs = get_paragraphs_of_article(article_url)
                # save_paragraphs(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n")