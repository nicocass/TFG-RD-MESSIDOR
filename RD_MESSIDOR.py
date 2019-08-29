#LIBRERÍAS EMPLEADAS#
import cv2
import pandas as pd
import numpy as np
import itertools
import json
import utils
import hep
import os
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern, greycomatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#DEFINICIÓN DE FUNCIONES#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

def extract_bv(image):
    """Extracción de la máscara de la red vascular.

        Parameters
        ----------
        image: array
            Imagen de fondo de ojo en escala de grises.

        Returns
        -------
        bv: array
            Máscara resultante con la red vascular correspondiente.
    """
    sizes = [5, 11, 23]		
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    aux1 = contrast_enhanced_green_fundus.copy()
    
    for w in sizes:
        aux2 = cv2.morphologyEx(aux1, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w,w)),
                              iterations = 1)
        aux1 = cv2.morphologyEx(aux2, cv2.MORPH_CLOSE, 
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w,w)),
                              iterations = 1)
    
    r1 = cv2.subtract(aux1, contrast_enhanced_green_fundus)
    r2 = clahe.apply(r1)		

    ret,r3 = cv2.threshold(r2, 15, 255, cv2.THRESH_BINARY)	
    mask = np.ones(r2.shape[:2], dtype = "uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(r3.copy(), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
			
    im = cv2.bitwise_and(r2, r2, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
                       iterations=1)	

    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(green_fundus.shape[:2], dtype = "uint8") * 255
    x1, xcontours, xhierarchy = cv2.findContours(
            fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)	
    
    for cnt in xcontours:
        shape = "indef"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circulo"	
        else:
            shape = "venas"
        if(shape == "circulo"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    bv = cv2.bitwise_not(finimage)
    bv = cv2.erode(bv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)),
                   iterations=1)
    bv = cv2.dilate(bv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)),
                               iterations=1)	
    
    return bv


def extract_od(im_in):
    """Extracción de la máscara del disco óptico.
    
        References
        ----------
        [1] D. Marin, A. Aquino, M. E. Gegundez-Arias and J. M. Bravo, 
        "Detecting the Optic Disc Boundary in Digital Fundus Images
        Using Morphological, Edge Detection, and Feature Extraction Techniques."
        doi: 10.1109/TMI.2010.2053042
        
        Parameters
        ----------
        im_in: array
            Máscara binaria con el contorno del disco óptico.

        Returns
        -------
        im_out: array
            Máscara resultante con el disco óptico correspondiente.
            
    """
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_out = im_th | im_floodfill
    
    return im_out


def pre_process(img, path):
    """Preprocesado de las imágenes de fondo de ojo y obtención de las máscaras.
    Extracción del canal verde, eliminación del ruido y realce del contraste.
    Obtención de las máscaras.
    
        Parameters
        ----------
        img: array
            Array de la imagen de fondo de ojo en escala de grises.
            
        path: string
            Dirección en la que se almacena cada imagen.

        Returns
        -------
        result: array
            Array de la imagen en escala de grises tras el preprocesado.
            
        mask: array
            Máscara binaria que contiene la red vascular, el disco óptico 
            y el background.
    """
    fundus = cv2.imread(path + '\\' + img)		
    b, green_fundus, r = cv2.split(fundus)
    
    #Cálculo de las máscaras.
    vessels = extract_bv(fundus)
    pathFolder2 = os.path.join(path, '{}'.format('contours'))
    contour = cv2.imread(pathFolder2 + '/' + img, cv2.IMREAD_GRAYSCALE)
    disc = extract_od(contour)
    ret, mask = cv2.threshold(green_fundus, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, 
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                     iterations = 1)
    mask = cv2.dilate(mask, 
                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                 iterations = 1)
    mask = cv2.bitwise_and(mask, vessels)
    mask = cv2.bitwise_and(mask, disc) 
    
    #Eliminación del ruido
    blurred = cv2.medianBlur(green_fundus, 5)
    
    #Ecualización del histograma para el realce del contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    result = clahe.apply(blurred)
    
    return result, mask


def rgb_preprocess(img, path):
    """Preprocesado de las imágenes de fondo de ojo en color.
    
        Parameters
        ----------
        img: 3D array
            Array de la imagen de fondo de ojo en color.
            
        path: string
            Dirección en la que se almacena cada imagen.

        Returns
        -------
        result: 3D array
            Array de la imagen en color tras el preprocesado.
    
    """
    image = cv2.imread(path + '\\' + img).astype(np.uint8)
    
    #Conversión al espacio de color HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #Eliminación del ruido
    image_hsv[:,:,0] = cv2.medianBlur(image_hsv[:,:,0], 5)
    image_hsv[:,:,1] = cv2.medianBlur(image_hsv[:,:,1], 5)
    image_hsv[:,:,2] = cv2.medianBlur(image_hsv[:,:,2], 5)
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    #Conversión al espacio de color YUV
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    
    #Ecualización del histograma para el realce del contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    image_yuv[:,:,0] = clahe.apply(image_yuv[:,:,0])
    result = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    
    return result


def descriptor(img, descr):
    """Cálculo de los vectores de características para las imágenes 
    en escala de grises a través de distintos descriptores de textura.
    
        Parameters
        ----------
        img: 1D array
            Array de la imagen de fondo de ojo en escala de grises.
            
        descr: string
            Nombre del descriptor utilizado. En el se incluye la información
            del número de píxeles del vecindario y la escala utilizados.

        Returns
        -------
        caract: array
            Vector de características de dimensión variable en función 
            del descriptor y sus parámetros.
    
    """
    listofdescr = {'glcm_def': greycomatrix(img_as_ubyte(img),
                                            distances = distances,
                                            angles = angles,
                                            symmetric = True,
                                            normed = True),
    'glcm_ri': greycomatrix(img_as_ubyte(img),
                             distances = distances,
                             angles = angles,
                             symmetric = True,
                             normed = True),
    'lbp3x3_u2':local_binary_pattern(img, 8, 1, method='nri_uniform'),
    'lbp5x5_u2':local_binary_pattern(img, 16, 2, method='nri_uniform'),
    'lbp7x7_u2':local_binary_pattern(img, 24, 3, method='nri_uniform'),
    'lbp3x3_def':local_binary_pattern(img, 8, 1, method='default'),
    'lbp5x5_def':local_binary_pattern(img, 16, 2, method='default'),
    'lbp7x7_def':local_binary_pattern(img, 24, 3, method='default'),
    'lbp3x3_ri':local_binary_pattern(img, 8, 1, method='ror'),
    'lbp5x5_ri':local_binary_pattern(img, 16, 2, method='ror'),
    'lbp7x7_ri':local_binary_pattern(img, 24, 3, method='ror'),
    'lbp3x3_riu2':local_binary_pattern(img, 8, 1, method='uniform'),
    'lbp5x5_riu2':local_binary_pattern(img, 16, 2, method='uniform'),
    'lbp7x7_riu2':local_binary_pattern(img, 24, 3, method='uniform')}

    caract = listofdescr[descr]
    
    return caract


def ics(img, descr):
    """Cálculo de los vectores de características para las imágenes 
    en color a través del descriptor ICS-LBP con orden cromático.
    
        Parameters
        ----------
        img: 1D array
            Array de la imagen de fondo de ojo en color.
            
        descr: string
            Nombre del descriptor utilizado. En el se incluye la información
            del número de píxeles del vecindario, la escala utilizados,
            y el orden cromático.

        Returns
        -------
        caract: array
            Vector de características de dimensión variable en función 
            del descriptor y sus parámetros.
    
    """
    rad = int(descr[-1:])
    
    if descr == 'icslbpprod1':
        params = dict(order='product', radius=[rad])
    elif descr == 'icslbpRGB1':
        params = dict(order='lexicographic', radius=[rad], bands='RGB')     
    elif descr == 'icslbpGRB1':
        params = dict(order='lexicographic', radius=[rad], bands='GRB')
    elif descr == 'icslbpGBR1':
        params = dict(order='lexicographic', radius=[rad], bands='GBR')
    elif descr == 'icslbpprod2':
        params = dict(order='product', radius=[rad])
    elif descr == 'icslbpRGB2':
        params = dict(order='lexicographic', radius=[rad], bands='RGB')
    elif descr == 'icslbpGRB2':
        params = dict(order='lexicographic', radius=[rad], bands='GRB')
    elif descr == 'icslbpGBR2':
        params = dict(order='lexicographic', radius=[rad], bands='GBR')
    elif descr == 'icslbpprod3':
        params = dict(order='product', radius=[rad])
    elif descr == 'icslbpRGB3':
        params = dict(order='lexicographic', radius=[rad], bands='RGB')
    elif descr == 'icslbpGRB3':
        params = dict(order='lexicographic', radius=[rad], bands='GRB')
    elif descr == 'icslbpGBR3':
        params = dict(order='lexicographic', radius=[rad], bands='GBR')
    result = hep.ImprovedCenterSymmetricLocalBinaryPattern(**params)
    code = result.codemap(img, rad, 2**rad)
    
    return code, rad


def graycoprops(P, prop):
    """Cálculo de las características de Haralick normalizadas 
    de cada una de las imágenes.
    
        Parameters
        ----------
        P: 1D array
            Matriz GLCM para una determinada orientación y desplazamiento.
            
        prop: string
            Nombre de la característica a calcular.

        Returns
        -------
        result: array
            Vector de características de dimensión variable en función 
            del descriptor y sus parámetros.
    
    """
    # Normaliza la matriz GLCM
    GLCM = P/P.sum()
    G = P.shape[0]
    
    u, v = np.indices(GLCM.shape)
    mu_u = np.sum(u*GLCM)
    mu_v = np.sum(v*GLCM)
    sigma_u = np.sqrt(np.sum((u - mu_u)**2 * GLCM))
    sigma_v = np.sqrt(np.sum((v - mu_v)**2 * GLCM))
    
    # Cálculo de las propiedades para cada GLCM    
    if prop == 'contrast':
       result = np.sum(GLCM*(u - v)**2)/(G - 1)**2 
    elif prop == 'correlation':
       result = .5 + np.sum((u - mu_u)*(v - mu_v)*GLCM)/(2*sigma_u*sigma_v) 
    elif prop == 'energy':
       result = np.sum(GLCM**2) 
    elif prop == 'entropy':
       result = -np.sum(GLCM*np.log2(GLCM + (GLCM==0)))/(2*np.log2(G)) 
    elif prop == 'homogeneity':
       result = np.sum(GLCM/(1 + np.absolute(u - v))) 
    else:
        raise ValueError('%s is an invalid property' % (prop))
    
    return result


def grid_search_cv(X, y, test_size, clf, params, rs):
    """Realiza una búsqueda exhaustiva de los los hyperparámetros óptimos 
    usado cross-validation. Con estos mismos realiza la predicción
    para el conjunto test.
    
    Parameters
    ----------
    X: array
        Vector de características.
    y: array
        Etiquetas de clase.
    clf: class
        Clase que implementa un clasificador.
    param_grid : dict
        Diccionario con los nombres de los parámetros (string) como clave y 
        lista de los posibles valores de los parámetros, los que definen la
        búsqueda exhaustiva llevada a cabo por `GridSearchCV`.
    n_folds : int
        Número de folds usados en el cross-validation. Al menos 2.
    test_size : float
        Proporción de muestras usadas en el test. Rango de valores entre 0 y 1.
    random_state : int
        Semilla del generador de números aleatorios. Determina la división 
        entre train y test, y los folds para el cross-validation.
    Returns
    -------
    gscv: array
        Características calculadas. El número de filas es igual al número de 
        muestras y el número de columnas es igual al espacio dimensional del 
        vector de características.
        
    test_score: array
        Predicciones para cada una de las muestras del conjunto test utilizando
        los parámetros óptimos del clasificador. 
    
    conf_matrix: array
        Matriz de confusión extraída del test_score. 
    """
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y)
    
    gscv = GridSearchCV(estimator = clf(),
                        cv = StratifiedKFold(n_splits=N_FOLDS,
                                             random_state=0), 
                        param_grid=params,
                        return_train_score=False)
    
    gscv.fit(X_train, y_train)
    best_clf = clf(**gscv.best_params_)
    best_clf.fit(X_train, y_train)
    test_score = best_clf.score(X_test, y_test)
    y_pred = best_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return [gscv, test_score, conf_matrix]

def feats_product(dS, pS, desc):
    """Cálculo de las posibles combinaciones a niel de escala
    de los descriptores LBP.
    
        Parameters
        ----------
        dS: string
            Nombre de las bases de datos.
            
        pS: string
            Dirección del archivo del vector de características .

        desc: list
            Lista con los nombres de los descriptor a combinar.
            
        Returns
        -------
        desc: list
            Lista con las posibles combinaciones de descriptores.
            
    """
    final_props = desc.copy()
    for dbase in dS:
        for feat_1, feat_2 in itertools.combinations(final_props, 2):
            
            feats_path = os.path.join(pS,'{}_{}{}.pkl'
                                      .format(dbase, feat_1, feat_2))
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            desc.append(feat_1 + feat_2)
            
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                combi = np.hstack((element1,element2))
                utils.save_object(combi, feats_path)
                
        for feat_1, feat_2, feat_3 in itertools.combinations(final_props, 3):
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            path_3 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_3))
            
            if feat_1 == 'glcm_defcontrast1glcm_defcorrelation1glcm_defenergy1glcm_defentropy1glcm_defhomogeneity1':
                feats_path = os.path.join(pS,'{}_glcm_def123.pkl'
                                          .format(dbase))
                desc.append('glcm_def123')
                
            elif feat_1 == 'glcm_ricontrast1glcm_ricorrelation1glcm_rienergy1glcm_rientropy1glcm_rihomogeneity1':
                feats_path = os.path.join(pS, '{}_glcm_ri123.pkl'
                                          .format( dbase))
                desc.append('glcm_ri123')
                
            else:
                feats_path = os.path.join(pS,'{}_{}{}{}.pkl'
                                          .format(dbase, feat_1, feat_2, feat_3))
                desc.append(feat_1 + feat_2 + feat_3)
                
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                element3 = utils.load_object(path_3)
                combi = np.hstack((element1,element2,element3))
                utils.save_object(combi, feats_path)
                
    return desc

def haralick_product(dS, pS, desc):
    """Cálculo de todas las posibles combinaciones para las 5 características
    de Haralick escogidas.
        
        Parameters
        ----------
        dS: string
            Nombre de las bases de datos.
            
        pS: string
            Dirección del archivo del vector de características .

        desc: list
            Lista con los nombres de los descriptor a combinar.
            
        Returns
        -------
        desc: list
            Lista con las posibles combinaciones de descriptores.
            
    """
    final_props = desc.copy()
    for dbase in dS:
        for feat_1, feat_2 in itertools.combinations(final_props, 2):
            feats_path = os.path.join(pS,'{}_{}{}.pkl'
                                      .format(dbase, feat_1, feat_2))
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            desc.append(feat_1+feat_2)
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                combi = np.hstack((element1,element2))
                utils.save_object(combi, feats_path)
                
        for feat_1, feat_2, feat_3 in itertools.combinations(final_props, 3):
            feats_path = os.path.join(pS,'{}_{}{}{}.pkl'
                                      .format(dbase, feat_1,
                                              feat_2, feat_3))
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            path_3 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_3))
            desc.append(feat_1 + feat_2 + feat_3)
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                element3 = utils.load_object(path_3)
                combi = np.hstack((element1,element2,element3))
                utils.save_object(combi, feats_path)
            
        for feat_1, feat_2, feat_3, feat_4 in itertools.combinations(final_props, 4):
            feats_path = os.path.join(pS,'{}_{}{}{}{}.pkl'
                                      .format(dbase, feat_1,
                                              feat_2, feat_3,
                                              feat_4))
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            path_3 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_3))
            path_4 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_4))
            desc.append(feat_1 + feat_2 + feat_3 + feat_4)
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                element3 = utils.load_object(path_3)
                element4 = utils.load_object(path_4)
                combi = np.hstack((element1,element2,element3,element4))
                utils.save_object(combi, feats_path)
            
        for feat_1, feat_2, feat_3, feat_4, feat_5 in itertools.combinations(final_props, 5):
            feats_path = os.path.join(pS,'{}_{}{}{}{}{}.pkl'
                                      .format(dbase, feat_1,
                                              feat_2, feat_3,
                                              feat_4, feat_5))
            path_1 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_1))
            path_2 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_2))
            path_3 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_3))
            path_4 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_4))
            path_5 = os.path.join(pS, '{}_{}.pkl'.format(dbase, feat_5))
            desc.append(feat_1 + feat_2 + feat_3 + feat_4 + feat_5)
            if not os.path.isfile(feats_path):
                element1 = utils.load_object(path_1)
                element2 = utils.load_object(path_2)
                element3 = utils.load_object(path_3)
                element4 = utils.load_object(path_4)
                element5 = utils.load_object(path_5)
                combi = np.hstack(
                        (element1, element2, element3, element4, element5))
                utils.save_object(combi, feats_path)

    return desc

def split_list(lista, divisor):
    div = len(lista)//len(divisor)
    return lista[:div], lista[div:div*2], lista[div*2:]
    
def calc_haralick(pro, dist, dBase, pSave, descript):    
    f_desc = []
    for i, d in enumerate(dist):
        for j, p in enumerate(pro):
            f_desc.append(descript + p + str(d))
    splitted = list(split_list(f_desc, dist))
    f_desc.clear()
    f_desc = [haralick_product(
            dBase, pSave, splitted[i]) for i, n in enumerate(splitted)]
    f_desc = list(itertools.chain.from_iterable(f_desc))
    return f_desc


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#CONFIGURACIÓN DEL MODELO #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    
pathSave = "D:\resultados"
pathImages = "D:\resultados\images"
pathMasks = "D:\resultados\masks"
pathRGB = "D:\resultados\RGB"
pathFolder = "D:\ophthalmologic-datasets\MESSIDOR"

datasets = ['MESSIDOR']
folders = ['Base11','Base12','Base13','Base14',
        'Base21', 'Base22','Base23','Base24',
        'Base31','Base32','Base33','Base34']

nimages = 1200
N_FOLDS = 5
n_tests = 5
test_size = 1/4

descriptors = ['glcm_def', 'glcm_ri',
        'lbp3x3_def', 'lbp5x5_def', 'lbp7x7_def',
        'lbp3x3_u2','lbp5x5_u2','lbp7x7_u2',
        'lbp3x3_ri', 'lbp5x5_ri','lbp7x7_ri',
        'lbp3x3_riu2', 'lbp5x5_riu2', 'lbp7x7_riu2',
        'icslbpprod1','icslbpprod2','icslbpprod3',
        'icslbpGRB1','icslbpGRB2','icslbpGRB3',
        'icslbpGBR1','icslbpGBR2','icslbpGBR3',
        'icslbpRGB1','icslbpRGB2','icslbpRGB3']

props = ['contrast', 'correlation',
        'energy', 'entropy', 'homogeneity']

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

size = {
        'glcm_def': len(angles),
        'glcm_ri': len(angles)-1,
        'lbp3x3_u2':59,'lbp5x5_u2':243,'lbp7x7_u2':555,
        'lbp3x3_def':2**8,'lbp5x5_def':2**16,'lbp7x7_def':2**12,
        'lbp3x3_ri':2**8,'lbp5x5_ri':2**16,'lbp7x7_ri':2**12,
        'lbp3x3_riu2':10,'lbp5x5_riu2':18,'lbp7x7_riu2':26,
        'icslbpRGB1':16,'icslbpprod1':16,'icslbpGBR1':16,'icslbpGRB1':16,
        'icslbpRGB2':256,'icslbpprod2':256,'icslbpGBR2':256,'icslbpGRB2':256,
        'icslbpRGB3':4096,'icslbpprod3':4096,'icslbpGBR3':4096,'icslbpGRB3':4096
        }

estimators = [
        (KNeighborsClassifier, dict(n_neighbors = range(1, 9, 2))),
        (SVC, dict(C=[2**i for i in range(-5, 15, 2)],
                      gamma=[2**j for j in range(-15, 3, 2)]))
        ] 
 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#LECTURA DEL DATASET MESSIDOR Y ETIQUETADO DE LAS IMÁGENES#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

carpeta=os.path.join(pathFolder, '{}'.format('Base11'))
data = pd.read_excel(os.path.join(
        carpeta, 'Annotation_{}.xls'.format('Base11')))
imgs = pd.Series(data['Image name'])
y = pd.Series(np.int_(data['Retinopathy grade']) > 0).values

for folder in folders[1:]:
    carpeta = os.path.join(pathFolder, '{}'.format(folder))
    data = pd.read_excel(os.path.join(carpeta, 'Annotation_{}.xls'
                                      .format(folder)))
    imgs = imgs.append(pd.Series(data['Image name']))
    serie = pd.Series(np.int_(data['Retinopathy grade']) > 0).values
    y = np.hstack((y, serie))


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#CÁLCULO DE LAS CARACTERÍSTICAS CON DISTINTOS DESCRIPTORES
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

for dbase, descr in itertools.product(datasets, descriptors):
    feats_path = os.path.join(pathSave, '{}_{}.pkl'.format(dbase, descr))
    count = 0
    if not os.path.isfile(feats_path):
        
        X = np.zeros(shape = (nimages, size[descr]), dtype = np.float64)
        
        if descr == 'glcm_ri':
            feats = np.zeros(shape=(len(props), len(angles)))
            feats2 = np.zeros(shape=(len(props), len(angles)-1))
            dist_feats = np.zeros(
                    shape=(nimages, len(distances), len(props),len(angles)-1),
                    dtype=np.float64)
            
        elif descr == 'glcm_def':
           feats = np.zeros(shape=(len(props), len(angles)))
           dist_feats = np.zeros(
                   shape=(nimages, len(distances),len(props), len(angles)),
                   dtype=np.float64)
           
        for folder in folders:
            carpeta = os.path.join(pathFolder, '{}'.format(folder))
            filesArray = []
            
            with os.scandir(carpeta) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.tif'):
                        filesArray.append(entry.name)
                        
            for file_name in filesArray:
                i_path = os.path.join(pathImages, '{}'.format(file_name))
                m_path = os.path.join(pathMasks, '{}'.format(file_name))
                rgb_path = os.path.join(pathRGB, '{}'.format(file_name))
                if not os.path.isfile(i_path):
                    contrast_enhanced_green_fundus, mask = pre_process(
                            file_name, carpeta)
                    utils.save_object(contrast_enhanced_green_fundus, i_path)
                    utils.save_object(mask, i_path)
                if not os.path.isfile(rgb_path):
                    rgb = rgb_preprocess(file_name, carpeta)
                    utils.save_object(rgb, rgb_path)
                else:
                    contrast_enhanced_green_fundus = utils.load_object(i_path)
                    mrgb = utils.load_object(rgb_path)
                    mask = utils.load_object(m_path)
          
                if descr in ['glcm_def', 'glcm_ri']:  
                        mask_inv = cv2.bitwise_not(mask)
                        masked = contrast_enhanced_green_fundus.copy()
                        masked[masked == 0] = 1
                        masked[mask_inv > 0] = 0
                        glcm = descriptor(masked, descr)
                        glcm = glcm[1:, 1:, :, :]    
                        for i, d in enumerate(distances):
                            for j, a in enumerate(angles):
                                for k, p in enumerate(props):
                                    feats[k, j] = graycoprops(
                                            glcm[:, :, i, j], p)
                            if descr == 'glcm_ri':
                                rifeats = np.absolute(np.fft.fft(feats))
                                feats2 = rifeats[:, :-1]                
                                for k, p in enumerate(props):
                                    dist_feats[count, i, k, :] = feats2[k, :]
                            else:            
                                for k, p in enumerate(props):
                                    dist_feats[count, i, k, :] = feats[k, :]
                                    
                elif descr in  ['icslbpprod1','icslbpprod2','icslbpprod3',
                                'icslbpGRB1','icslbpGRB2','icslbpGRB3',
                                'icslbpGBR1','icslbpGBR2','icslbpGBR3',
                                'icslbpRGB1','icslbpRGB2','icslbpRGB3']:
                    icslbp, r = ics(mrgb, descr)
                    icslbp = icslbp.astype(np.uint8)
                    mask_inv = cv2.bitwise_not(mask)
                    maskk = cv2.dilate(mask_inv,
                                       cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                 (3,3)),
                                                                 iterations = 1)
                    mask = cv2.bitwise_not(maskk)
                    cropped = mask[r:-r,r:-r]
                    hist_mask = cv2.calcHist([icslbp], [0],
                                             cropped, [size[descr]],
                                             [0,size[descr]])
                    hist_mask = hist_mask.ravel()
                    X[count,:] = np.true_divide(hist_mask, hist_mask.sum(axis=0))

                else:
                    lbp = descriptor(contrast_enhanced_green_fundus, descr)
                    lbp = lbp.astype(np.uint8)
                    mask_inv = cv2.bitwise_not(mask)
                    maskk = cv2.dilate(mask_inv,
                                       cv2.getStructuringElement(
                                               cv2.MORPH_RECT, (3, 3)),
                                               iterations = 1)
                    mask = cv2.bitwise_not(maskk)
                    hist_mask = cv2.calcHist([lbp], [0],
                                             mask, [size[descr]],
                                             [0,size[descr]])
                    hist_mask = hist_mask.ravel()
                    X[count,:] = np.true_divide(
                            hist_mask, hist_mask.sum(axis=0))
                count+=1
                
        haralick = []
        
        if descr in ['glcm_def',  'glcm_ri']:
            for i, d in enumerate(distances):
                for j, p in enumerate(props):
                    feats_path = os.path.join(pathSave, '{}_{}{}{}.pkl'
                                              .format(dbase, descr, p, str(d)))
                    haralick.append(descr + p + str(d))
                    X = dist_feats[:, i, j, :]
                    utils.save_object(X, feats_path)
                    
        else:    
            utils.save_object(X, feats_path)
            
            
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#CONCATENACIÓN MULTIESCALA DE LOS DESCRIPTORES#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

final_desc = []

final_desc.append(
        calc_haralick(props, distances, datasets, pathSave, descriptors[0]))
final_desc.append(
        calc_haralick(props, distances, datasets, pathSave, descriptors[1]))
final_desc.append(feats_product(datasets, pathSave, descriptors[2:5]))
final_desc.append(feats_product(datasets, pathSave, descriptors[5:8]))
final_desc.append(feats_product(datasets, pathSave, descriptors[8:11]))
final_desc.append(feats_product(datasets, pathSave, descriptors[11:14]))
final_desc.append(feats_product(datasets, pathSave, descriptors[14:17]))
final_desc.append(feats_product(datasets, pathSave, descriptors[17:20]))
final_desc.append(feats_product(datasets, pathSave, descriptors[23:26]))
final_desc.append(feats_product(datasets, pathSave, descriptors[26:]))
final_desc.append(feats_product(datasets, pathSave, final_desc[0][30:31*3:31]))
final_desc.append(feats_product(datasets, pathSave, final_desc[1][30:31*3:31]))

final_desc = [item for sublist in final_desc for item in sublist]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#CLASIFICACIÓN: TRAIN Y TEST#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

count = 0
cols = ['TP', 'FP', 'FN', 'TN']
rows = []
data = np.zeros(shape = (len(estimators)*len(final_desc)*5, len(cols)))

for prod_tup in itertools.product(datasets, final_desc, estimators):
    dbase, descr, (clf, cv_param) = prod_tup
    result_path = os.path.join(pathSave,'{}_{}_{}.pkl'
                               .format(dbase, descr, clf.__name__))
    
    if os.path.isfile(result_path):
        result = utils.load_object(result_path)
        
    else:
       feats_path = os.path.join(pathSave,
                                 '{}_{}.pkl'.format(dbase, descr))
       
       X = utils.load_object(feats_path)
       result = [grid_search_cv(X, y, test_size, clf, cv_param, n)
                 for n in range(n_tests)]
       
       utils.save_object(result, result_path)
       
    if result is not None:
        print('RESULTADOS: {} {}'
              .format(descr, clf.__name__))
        print('Mean best cv score: {:.2f}'.format(
            100*np.mean([g.best_score_ for g, ts, cm in result])))
        print('Mean test score: {:.2f}'
              .format(100*np.mean([ts for g, ts, cm in result])))
        print('Confusion Matrix: {}'
              .format([result[i][-1] for i, n in enumerate(result)]))
        rows.append(f'{descr} {clf.__name__}')
        
        for gscv, score, cm in result:
            rows.append(json.dumps(gscv.best_params_))
        count += 1
    
        for i, n in enumerate(result):
            for j in range(2):
                for k in range(2):
                    data[count, j*2 + k ] = result[i][-1][j,k]
            count += 1  
    
    for gscv, score, cm in result:
        print(gscv.best_params_, score)
    print('\n')
#Exportar los resultados a un fichero .xlsx    
df = pd.DataFrame(data, index = rows, columns = cols)
df.to_excel('RESULTADOS.xlsx')