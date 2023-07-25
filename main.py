#__________________________________________import des modules___________________________________________________________

import csv
import cv2
import mediapipe as mp
from copy import deepcopy

from model.classification import classifier

#__________________________________________import des fonctions___________________________________________________________


def calc_landmarks_list(image,hand):
    """Calcule la liste des points landmarks de la main détectée
    Paramètres:
        image: image de la webcam
        hand: main détectée
    Retourne:
        landmark_list: liste des points landmarks de la main détectée avec un tuple (cx, cy) pour chaque point
        cx est la coordonnée x du point landmark
        cy est la coordonnée y du point landmark
    Remarques :
        on ne garde pas cz (la profondeur) car cela ne nous interesse pas
        id est l'identifiant du point landmark (de 0 à 20) (cf MediaPipe Hands), on ne le garde pas non plus dans la liste pour mieux gérer la normalisation après
    """
    image_width,image_height=image.shape[1],image.shape[0]
    landmark_list=[]

    for id,lm in enumerate(hand.landmark):
        cx,cy=int(lm.x*image_width),int(lm.y*image_height)
        landmark_list.append([cx,cy])
    return landmark_list


def calc_bouding_rect(image,hand):
    """Calcule la bouding box de la main détectée (rectangle qui englobe la main, utile pour plus tard)
    Paramètres:
        image: image de la webcam
        hand: main détectée
    Retourne:
        brect: tuple (xmin,ymin,width,height) de la bouding box de la main détectée
    """
    image_width,image_height=image.shape[1],image.shape[0]
    x_list=[]
    y_list=[]
    for id,lm in enumerate(hand.landmark):
        cx,cy=int(lm.x*image_width),int(lm.y*image_height)
        x_list.append(cx)
        y_list.append(cy)
    xmin,xmax=min(x_list),max(x_list)
    ymin,ymax=min(y_list),max(y_list)
    brect=xmin,ymin,xmax-xmin,ymax-ymin
    return brect


def process_landmarks_lists(landmarks_list):
    """Normalise les coordonnées des points landmarks de la main détectée
    Paramètres:
        landmarks_list: liste des points landmarks de la main détectée avec un tuple (cx, cy) pour chaque point
        cx est la coordonnée x du point landmark
        cy est la coordonnée y du point landmark
    Retourne:
        temp_landmarks_list: liste des points landmarks de la main détectée normalisés avec un tuple (cx, cy) pour chaque point
    """
    temp_landmarks_list=deepcopy(landmarks_list)
    base_x=temp_landmarks_list[0][0]
    base_y=temp_landmarks_list[0][1]
    max_value=0     #initialisation de la valeur max
    for i in range(len(temp_landmarks_list)):
        #on soustraie les coordonnées de l'origine pour avoir un repère centré sur la main (0,0 au point 0 de MediaPipe Hands, c'est à dire la base du poignet)
        temp_landmarks_list[i][0]=landmarks_list[i][0]-base_x
        temp_landmarks_list[i][1]=landmarks_list[i][1]-base_y
        max_temp=max(abs(temp_landmarks_list[i][0]),abs(temp_landmarks_list[i][1])) 
        max_value=max(max_value,max_temp)       #on sauvegarde la valeur max (valeur absolue) pour la normalisation

    #normalisation des coordonnées des points landmarks
    for i in range(len(temp_landmarks_list)):
        temp_landmarks_list[i][0]=temp_landmarks_list[i][0]/max_value
        temp_landmarks_list[i][1]=temp_landmarks_list[i][1]/max_value
    
    #enlever les doubles crochets de la liste pour avoir une liste (et non une liste de liste) => nécessaire pour avoir le bon format pour le csv et pour la classification
    new_landmarks_list=[]
    for i in range(len(temp_landmarks_list)):
        new_landmarks_list.append(temp_landmarks_list[i][0])
        new_landmarks_list.append(temp_landmarks_list[i][1])

    return new_landmarks_list


def logging_csv(number,landmarks_list,mode):
    """Ecriture des points landmarks de la main détectée dans un fichier csv
    Paramètres:
        number: numero de label (le signe en question de la main)
        landmarks_list: liste des points landmarks de la main détectée avec un tuple (cx, cy) pour chaque point (cf landmarks de Mediapipe (0 à 20))
    """
    if mode==0:
        pass
    if mode==1 and number>=0:       #vérification, ne pas enregistrer si le numéro de label est négatif, cad si on a pas appupyé sur p
        print("ecriture du dataset")
        csv_path=".\model\dataset.csv"        #relative path
        with open(csv_path,'a',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([number, *landmarks_list])
            
            # dans le cas on on avait une liste de liste au lieu d'une liste simple, on aurait fait :
            # writer.writerow([number] + [float(coord) for point in landmarks_list for coord in point])



def select_save(key,mode,label):
    """Demander à enregistrer le signe de la main (label)
    Paramètres:
        key: touche pressée
        mode: mode actuel (0=normal, 1=apprentissage) => permet de vérifier de ne pas enregistrer en mode normal
        number : numero de label
    Retourne:
        number: numero de label, utile pour le csv
    Remarques :
        on réinitialise le numero de label (à -1) à chaque fois que la fonction se lance"""
    number=-1
    if key==ord('p') and mode==1:
        print(f"demande enregistrement pour le label numero {label}")
        number=label

    return number

def draw_info(image,hand_sign_text,brect,score):
    """Affiche le signe de la main détecté
    Paramètres:
        image: image de la webcam
        hand_sign_text: texte du signe de la main détecté
        brect: bouding box de la main détectée
    Retourne:
        image : image de la webcam modifiée avec les infos ajoutées
    """
    score_text=str(round(score,1)) + "%"
    cv2.rectangle(image,(brect[0],brect[1]),(brect[0]+brect[2],brect[1]+brect[3]),color=(186,85,211),thickness=2)
    #rectangle plein pour le texte (pour que le texte soit plus lisible)
    cv2.rectangle(image,(brect[0],brect[1]-30),(brect[0]+len(hand_sign_text+score_text)*20,brect[1]),color=(186,85,211),thickness=-1)
    cv2.putText(image,hand_sign_text,(brect[0],brect[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,score_text,(brect[0]+28,brect[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    
    return image


def main():
    """Fonction principale du programme : dectection, enregistrement des données, affichage et reconnaissance des signes
    Le programme s'arrête en pressant la touche q
    L'apprentissage se fait dans un autre programme de ML
    Paramètres:
        aucun
    Retourne:
        rien
    """
    
    #initialisation de la webcam
    cap=cv2.VideoCapture(0)
    cap_width,cap_height=1200,1000
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cap_height)
    
    #initialisation du modèle de mediapipe
    mp_drawing=mp.solutions.drawing_utils
    mp_hands=mp.solutions.hands
    hands=mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        max_num_hands=2) 

    #initialisation du modèle de classification (reconnaissance de signe de la main)
    keypoint_classifier=classifier()

    #lecture des labels (permet de relier l'id du signe au véritable nom du signe (sorte de traduction))
    with open(".\model\label.csv",encoding="utf-8") as f:        #relative path
        keypoint_classification_label=csv.reader(f)
        #on met sous forme d'une liste
        keypoint_classification_label=[ 
            row[0] for row in keypoint_classification_label
        ]

    #initialisation du mode (0=normal, 1=apprentissage), par défaut on est en mode normal
    mode=0
    global label
    label=-1

    while cap.isOpened():       #tant que la webcam est ouverte
        ret,frame=cap.read()
        
        #fermeture de la fenetre en pressant la touche q
        if cv2.waitKey(10) & 0xFF==ord('q'):
            print("fin du programme")
            break
        
        #selection du mode en pressant les touches indiquées => par defaut, on est en mode "normal"
        key=cv2.waitKey(10)
        if key==ord('k'):
            mode=1
            print("mode apprentissage")
            label=int(input("identifiant du label: ?"))     #choix du label (cad du signe de la main) à enregistrer pour entrainer le modèle après
        if key==ord('l'):
            mode=0
            print("mode normal")

        #recupération du numéro de label pour l'écriture dans le dataset dans le cas du mode apprentissage
        number=select_save(key,mode,label)      

        #recoloration de l'image BGR en RGB (Mediapipe utilise le RGB)
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    
        
        #on inverse l'image pour pas avoir le coté "inversé" a cause de la webcam
        image=cv2.flip(image,1)
        
        #detection des mains avec mediapipe
        image.flags.writeable=False
        results=hands.process(image)
        image.flags.writeable=True

        #recoloration de l'image RGB en BGR
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        
        #rendu des resultats => affichage
        if results.multi_hand_landmarks:    #si la main est détectée    
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):   #on loop pour chaque main détectée
                
                #bouding box => utile pour plus tard
                brect=calc_bouding_rect(image,hand_landmarks)
                # cv2.rectangle(image,(brect[0],brect[1]),(brect[0]+brect[2],brect[1]+brect[3]),color=(186,85,211),thickness=1)
                
                #liste des points landmarks (utile pour la normalisation)
                landmarks_list=calc_landmarks_list(image,hand_landmarks)
                #liste des points landmarks normalisés (utile pour la reconnaissance)
                process_landmarks_list=process_landmarks_lists(landmarks_list)
               
                #écrire dans le dataset, ne se passe rieqqqn si mode normal
                logging_csv(number,process_landmarks_list,mode)
                        
                #on detecte main gauche ou main droite  
                handType=handedness.classification[0].label         
                if handType=="Left":        #vert pour la main gauche, rouge pour la main droite
                    color_type1=(0,255,0)
                    color_type2=(0,0,255)
                if handType=="Right":
                    color_type1=(0,0,255)
                    color_type2=(0,255,0)
                
                #dessin du "squelette" de la main (avec les points landmarks et distinction main gauche/droite)
                mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=color_type1,thickness=2,circle_radius=2),
                                          mp_drawing.DrawingSpec(color=color_type2,thickness=2,circle_radius=2))  #color=(B,G,R)
                
                #classification des signes de la main
                hand_sign_id=keypoint_classifier(process_landmarks_list)[0]
                hand_sign_text=keypoint_classification_label[hand_sign_id]
                score=(keypoint_classifier(process_landmarks_list)[1])*100
                
                draw_info(image,hand_sign_text,brect,score)
        
        #affichage du mode choisi      
        if mode==1:
            cv2.putText(image,"Mode apprentissage",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        if number!=-1:
            cv2.putText(image,"(P) Enregistrement du label "+str(number),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        
        cv2.imshow("Hand Gesture Recognition",image)        #affichage de l'image
               
    #libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


#__________________________________________________main_______________________________________________________________
if __name__=="__main__":
    main()
