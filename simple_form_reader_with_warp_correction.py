import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
#from metodos import warpImagem


numero_aluno=0

def warpImagem(im1, im2):

    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create(200)  # Registration works with at least 50 points


    # encontrar os pontos chave e descritores com o orb
    kp1, des1 = orb.detectAndCompute(img1, None)  # kp1 --> lista dos pontos chave
    kp2, des2 = orb.detectAndCompute(img2, None)

    print("//////////////// Orb 1", des1)
    print("//////////////// Orb 2", des2)

    # Brute-Force para fazer match entre os key points de uma imagem e da outra
    # criar objecto de match

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    print("//////////////// Matcher Descriptor", matcher)

    # Match descriptors.
    matches = matcher.match(des1, des2, None)  # Creates a list of all matches, just like keypoints

    # identificação das distancias
    matches = sorted(matches, key=lambda x: x.distance)
    print("//////////////// Matches entre as imagens", matches)


    # Desenhar os keupoints
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:30], None)
    cv2.imshow("Key Points Match entre foto e scan original", img3)

    # Now let us use these key points to register two images.
    # USar a homography.

    # Extrais locatização dos melhores matches
    # usar o RANSAC (RANSAC is )abbreviation of RANdom SAmple Consensus)


    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Usas homography

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))  # Applies a perspective transformation to an image.
    print("Estimated homography : \n", h)
    cv2.imshow("Imagem Corrigida", im1Reg)


    ########################################
    # Aplicar filtro de Ostu depois de passar para grayscale para treshold

    gray = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)

    global img_otsu
    ret, img_otsu = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Filtro Otsu", img_otsu)


    gray_scale = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
    img_bin = ~img_bin
    cv2.imshow("Filtro Aplicado", img_bin)

    ### selecting min size as 15 pixels
    line_min_width = 25


    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)

    ### Aplicar morfologia
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    # Merge das linhas verticais e horizontais
    img_bin_final = img_bin_h | img_bin_v
    cv2.imshow("Morfologia Binarização final", img_bin_h | img_bin_v)

    ### Usar component Connected Component Image e traçar zonas das checkboxes

    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    for x, y, w, h, area in stats[2:]:
        cv2.rectangle(im1Reg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(str(y)+" "+str(x)+" "+str(w)+" "+str(h))

    #####/////// primeiro Y depois X no crop////////

    ###checkbox 1
    crop_resposta_01 = im1Reg[394:394 + 47, 247:247 + 47]
    bin_crop_01= cv2.Canny(crop_resposta_01,100,200)
    count_pretos_01 = cv2.countNonZero(bin_crop_01)
    cv2.imshow("Cropped resposta 01", bin_crop_01)
    print("Numero de pixels pretos: "+str (count_pretos_01))

    ###checkbox 2
    crop_resposta_02 = im1Reg[452:452 + 47, 247:247 + 47]
    bin_crop_02 = cv2.Canny(crop_resposta_02, 100, 200)
    count_pretos_02 = cv2.countNonZero(bin_crop_02)
    cv2.imshow("Cropped resposta 01", bin_crop_02)
    print("Numero de pixels pretos: " + str(count_pretos_02))

    ###checkbox 3
    crop_resposta_03 = im1Reg[513:513 + 47, 247:247 + 47]
    bin_crop_03 = cv2.Canny(crop_resposta_03, 100, 200)
    count_pretos_03 = cv2.countNonZero(bin_crop_03)
    cv2.imshow("Cropped resposta 01", bin_crop_03)
    print("Numero de pixels pretos: " + str(count_pretos_03))

    ###checkbox 4
    crop_resposta_04 = im1Reg[570:570 + 47, 247:247 + 47]
    bin_crop_04 = cv2.Canny(crop_resposta_04, 100, 200)
    count_pretos_04 = cv2.countNonZero(bin_crop_04)
    cv2.imshow("Cropped resposta 01", bin_crop_04)
    print("Numero de pixels pretos: " + str(count_pretos_04))


    ######################################################
    ######################################################
    ### Calcular quais as respostas que estão assinaladas

    if count_pretos_01 > 180:
        respostaaluno=1
    if count_pretos_02 > 180:
        respostaaluno=2
    if count_pretos_03 > 180:
        respostaaluno=3
    if count_pretos_04 > 180:
        respostaaluno=4



    print("Valores do stats ::: "+str(stats[2:]))
    print("REsposta do aluno: ", respostaaluno)

    # Ler Ficheiro de Texto com as soluções
    file1 = open('resources/solution.txt', 'r')
    Respostas = file1.readlines()
    respostaPergunta1 = Respostas[0]
    print("Resposta Certa Pergunta 01 : ", respostaPergunta1)

    # Ler Ficheiro de Texto com as percentagens
    file2 = open('resources/score.txt', 'r')
    Percentagens = file2.readlines()
    scorePergunta1 = Percentagens[0]
    print("Pontuação Resposta 01 : ", scorePergunta1)

    global notafinal_aluno
    notafinal_aluno=0

    if str(respostaaluno) == str(respostaPergunta1):
        notafinal_aluno = str(scorePergunta1)
        print("Aluno empenhado Nota final :", str(notafinal_aluno))
    else:
        print("Aluno fraco Nota final :", str(notafinal_aluno))




    cv2.imshow("Imagem Original com o arranjo", im1Reg)

    # Aceder aos valores da matrix
    # The first cell is the number of labels
    num_labels = stats[0]
    # The second cell is the label matrix
    matrix = stats[1]
    # The third cell is the stat matrix
    stats = stats[2]
    # The fourth cell is the centroid matrix
    centroids = stats[3]



    print("/////////////// Valores da Matrix")
    print("Nº Labels: ", num_labels)
    print("Nº Matrix: ", matrix)
    print("Nº Stats: ", stats)





########################################
# Caminho das imagens
path_of_images = r"scans"
list_of_images = os.listdir(path_of_images)

#######################################
##### Ler imagem, aplicar warp e filtros

########################################
# Ciclo For para ler cada uma por ordem
#for image in list_of_images:

im2 = cv2.imread("images/template01.jpg")  # Imagem Template
#im1 = cv2.imread(os.path.join(path_of_images, image))  # Imagens distorcida
im1 = cv2.imread("scans/antoniocosta.jpg")

# Invocar metodo de processamento e identificação de contornos e passar imagens como parametros
warpImagem(im1, im2)


#########################################
######




##### Ler e gravar resultado

numero_aluno=numero_aluno+1
file3 = open('resources/pauta_final.txt', 'a')
file3.write('\n'+"Nota final de aluno Nº"+str(numero_aluno)+" : "+ str(notafinal_aluno))
file3.close()

cv2.waitKey()





#######################################
##### receber valores cotações
#lerScore()


# Fechar todas as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()























