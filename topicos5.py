
import csv

word_dict = {
     51: ' Casa ',
     52: ' Verão ',
     53: ' Jantar ',
     54: ' Filme ',
     55: ' Academia ',
     56: ' Família ',
     57: ' Festa ',
     58: ' Tarefa ',
     59: ' Café ',
     60: ' Presente ',
     61: ' Fome ',
     62: ' Música ',
     63: ' Cachorro ',
     64: ' Leitura ',
     65: ' Trabalho ',
     66: ' Restaurante ',
     67: ' Compras ',
     68: ' Parque ',
     69: ' Cinema ',
     70: ' Dia ',
     71: ' Academia ',
     72: ' Viagem ',
     73: ' Filme ',
     74: ' Flores ',
     75: ' Tarefa ',
     76: ' Amigos ',
     77: ' Compras ',
     78: ' Vista ',
     79: ' Bolo ',
     80: ' Tempo ',
     81: ' Churrasco ',
     82: ' Praia ',
     83: ' Filme ',
     84: ' Violão ',
     85: ' Museu ',
     86: ' Livro ',
     87: ' Dias ',
     88: ' Animais ',
     89: ' Natureza ',
     90: ' Jantar ',
     91: ' Férias ',
     92: ' Tempo ',
     93: ' Teatro ',
     94: ' Série ',
     95: ' Quarto ',
     96: ' Poesia ',
     97: ' Flores ',
     98: ' Cozinhar ',
     99: ' Parque ',
}


word_list = [{'word': word, 'id': _id} for _id, word in word_dict.items()]


# Abra o primeiro arquivo CSV para leitura
with open('primeiro_texto.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Pule o cabeçalho
   
    # Abra o segundo arquivo CSV para escrita
    with open('segundo_texto.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        # Escreva o cabeçalho no arquivo de saída
        writer.writerow(['source', 'target'])

        # Processar cada linha do primeiro arquivo CSV
        for row in reader:
            # Obtenha a frase e o ID da palavra
            frase = row[0].lower()                                                                                                                       
            for words in word_list:
                word_id = words['id']
                words = words['word'].lower()
                

                if words in frase:
                    writer.writerow([row[1], word_id])
                
           

print("Arquivo 'segundo_texto.csv' criado com sucesso.")
