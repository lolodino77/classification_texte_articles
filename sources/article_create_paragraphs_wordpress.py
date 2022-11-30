def create_paragraphs(self):
        """ Renvoie les paragraphes d'un article dans une liste
        
        Parametres: 
        article_url (string) : L'url de l'article a decouper en plusieurs parties
        
        Sortie:
        None : Fichier output_filename qui contient les documents de l'article dont l'url est article_url
        """
        # Recupere le texte de la page web a l'aide d'un parser
        # Recupere le texte d'un article mais avec des balises html (<\p><p> par exemple)
        page = requests.get(url=self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        txt = str(soup) 

        # Conversion des indicateurs de paragraphes et de sections /p et /li en retours a la ligne \n pour le split
        txt = txt.replace("\n", " ")
        txt = txt.replace("</p>", "</p>\n\n")
        txt = txt.replace("<li>", "<p>")
        txt = txt.replace("</li>", "</p>\n\n")
        
        # Suppression des balises html
        txt = html2text.html2text(txt)

        # Decoupage en plusieurs parties avec pour separateur le retour a la ligne \n
        paragraphs = txt.split("\n\n") 
        print("paragraphs =", paragraphs)

        #Enleve les doublons
        paragraphs = list(set(paragraphs))
        
        #Enleve les paragraphes qui contiennent # (car pas des paragraphes)
        paragraphs = [paragraphe for paragraphe in paragraphs if("#" not in paragraphe)]

        #Enleve les paragraphes qui contiennent un ou plusieurs adresses url
        paragraphs = [paragraphe for paragraphe in paragraphs if("http" not in paragraphe)]
   
        #Enleve les paragraphes avec trop peu de caracteres
        paragraphs = [paragraphe for paragraphe in paragraphs if len(paragraphe) > 12] 
        
        #Enleve les paragraphes avec des phrases trop courtes (trop peu de mots)
        paragraphs = [paragraphe for paragraphe in paragraphs if len(paragraphe.split(" ")) > 10]

        return(paragraphs)