import cv2
import numpy as np

############# PARÂMETROS #############
gaussian_m = 5                       # Linhas e colunas na matriz para gaussian blur.
escala = 0.5                         # Escala da primeira imagem que se pretende em relação à importada.
N_reducoes = 2                       # Número de reduções de cada imagem por piramidização.
######################################

################## IMAGENS A IMPORTAR E INFORMAÇÕES RELATIVAS ##########################

# Biblioteca com nomes dos ficheiros de imagens a importar e correspondente significado.
dict_sinais_identificar = {'pDireita': 'Direita',
                      'pEsquerda': 'Esquerda',
                      'pFrente': 'Frente',
                      'pMeta': 'Meta',
                      'pParque': 'Parque',
                      'pStop': 'Stop',
                      's01': 'Perigos Vários',
                      's11': 'Depressão',
                      's21': 'Animais',
                      's31': 'Passagem Estreita',
                      's02': 'Parque de Estacionamento',
                      's12': 'Limite de Velocidade Recomendado: 60',
                      's22': 'Hospital',
                      's32': 'Passadeira',
                      's03': 'Seguir para a Esquerda',
                      's13': 'Ligar Médios',
                      's23': 'Rotunda',
                      's33': 'Via para Transportes Públicos'
                      }

# Lista dos nomes dos ficheiros. É a lista de chaves para as bibliotecas que vão ser usadas.
lchave = []

# Lista com as imagens que vão ficar associadas a cada chave.
lvalor = []

# Lista com as dimensões de cada imagem: importadas e derivadas.
ldim = []

# Lista com o tipo de sinal de cada imagem. Função para proceder à distinção dos tipos de sinal através do nome.
ltipo = []

# Lista com a cor a ser associada à deteção de cada imagem.
lcor = []

# Lista da descrição do sinal.
ldescricao = []

# Função para proceder à distinção dos tipos de sinal através do seu nome, assim como da cor associada.
def def_tipo_cor(nome):
    tipo = ''
    cor = ''
    if nome[0] == 'p':
        tipo = 'Painel'
        cor = 'yellow'
    elif nome[0] == 's':
        if nome[2] == '1':
            tipo = 'Sinal de Perigo'
            cor = 'red'
        elif nome[2] == '2':
            tipo = 'Sinal de Informação'
            cor = 'green'
        elif nome[2] == '3':
            tipo = 'Sinal de Obrigação'
            cor = 'blue'
    return tipo, cor

# Biblioteca das cores a serem usadas.
dict_cores = { 'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255) }

# Criação das bibliotecas a serem usadas no processamento de vídeo.
for i in dict_sinais_identificar.keys():                           # Para cada nome de imagem da Biblioteca de sinais a identificar:
    chave = i + '_0'                                               # - Atribui o nome ... à primeira (original);
    valor = cv2.imread(i + '.png', cv2.IMREAD_GRAYSCALE)           # -- Importa a imagem correspondente em escala de cinzas;
    w = int(valor.shape[1] * escala)                               # -- Calcula as dimensões pretendidas para a imagem tendo em consideração a escala escolhida
    h = int(valor.shape[0] * escala)                               #   e que a original tem 700x700px. Nota:.shape = (altura , largura);
    dim = (w, h)                                                   # -- Compila as dimensões em "dim"
    valor = cv2.resize(valor, dim)                                 # -- Aplica as dimensões: dim = (largura,altura);
    tipo, cor = def_tipo_cor(i)                                    # -- Caracteriza o sinal quanto ao tipo e cor de deteção
    descricao = dict_sinais_identificar.get(i)                     # -- Guarda a descrição correspondente ao sinal

    lchave.append(chave)                                           # -- acrescenta chave à lista de chaves
    ldim.append(dim)                                               # -- acrescenta dim à lista de dimensões das imagens
    lvalor.append(valor)                                           # -- acrescenta valor à lista de valores(imagens)
    lcor.append(cor)                                               # -- acrescenta cor à lista de cores
    ltipo.append(tipo)                                             # -- acrescenta tipo à lista de tipos
    ldescricao.append(descricao)                                   # -- acrescenta descrição à lista de descrições

    for n in range(N_reducoes):                                    # - Para cada redução da original
        chave = i + '_' + str(n+1)                                 # -- Atribui nome
        valor = cv2.pyrDown(valor)                                 # -- Faz a respetiva redução
        w = valor.shape[1]                                         # -- Verifica as dimensões
        h = valor.shape[0]                                         #    e
        dim = (w, h)                                               #    grava-as em "dim"

        lchave.append(chave)                                       # -- acrescenta chave à lista de chaves
        ldim.append(dim)                                           # -- acrescenta dim à lista de dimensões das imagens
        lvalor.append(valor)                                       # -- acrescenta cor à lista de cores
        lcor.append(cor)                                           # -- acrescenta cor à lista de cores
        ltipo.append(tipo)                                         # -- acrescenta tipo à lista de tipos
        ldescricao.append(descricao)                               # -- acrescenta descrição à lista de descrições

for i in range(len(lchave)):                                                 # Para todas as imagens:
    lvalor[i] = cv2.GaussianBlur(lvalor[i], (gaussian_m, gaussian_m), 0)     # - aplica filtro que ajuda na deteção
    cv2.imshow(lchave[i], lvalor[i])                                         # - mostra a imagem

# Compactação em Bibliotecas do conteúdo criado:
dict_img = dict(zip(lchave, lvalor))
dict_dim = dict(zip(lchave, ldim))
dict_cor = dict(zip(lchave , lcor))
dict_tipo = dict(zip(lchave, ltipo))
dict_descricao = dict(zip(lchave, ldescricao))

############################################################################################################

############################## CAPTURA E PROCESSAMENTO DE VÍDEO ############################################

# Início da captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Ler cada frame
    _, frame = cap.read()
    # Criar frame em escala de cinzas
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Para cada imagem:
    for img in lchave:
        res = cv2.matchTemplate(gray_frame, dict_img.get(img), cv2.TM_CCOEFF_NORMED)    # Comparar a imagem com o frame
        loc = np.where(res >= 0.7)                                                      # Localizar correspondências
        w, h = dict_dim.get(img)                                                        # Ler dimensões da imagem a ser comparada
        for pt in zip(*loc[::-1]):                                                                         # Para cada correspondência encontrada:
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), dict_cores.get(dict_cor.get(img)), 3)         # - Desenhar retângulo com as dimensões lidas e cor atribuída
            print('Detetado: ' + img + '   --->   ' + dict_tipo.get(img) + ': ' + dict_descricao.get(img)) # - Texto a ser apresentado quando é encontrada correspondência
    # Mostrar frame com grafismo programado
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:           # Premir "ESC"
        break               # Terminar ciclo While

cap.release()               # Desligar captura de vídeo
cv2.destroyAllWindows()     # Fechar janelas

###########################################################################################################