from parser.parser import SpiceParser

parser = SpiceParser(lang="pt", wn_model="own-pt:1.0.0")

a = "A menina feliz entregou dois livros de romance para o amigo ontem."
b = "Os elefantes do circo foram libertados"
c = "Descrição: Fotografia. Uma pessoa com cabelo ruivo e uma tatuagem no nariz está fazendo uma pose com o dedo indicador levantado. Ela está vestindo uma jaqueta jeans e uma camiseta preta. No fundo, há uma prateleira com roupas. [Fim da descrição]"
graph = parser.parse(c)

print(graph.objects, graph.attributes, graph.relations)
