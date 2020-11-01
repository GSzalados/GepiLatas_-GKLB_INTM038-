import cv2
import numpy as np
import imutils
import pytesseract as tess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 
kernel = np.ones((1,1),np.uint8)

kep=cv2.imread(r"C:\Python\40.jpg") #---<-----------------------input

cv2.imwrite(r"C:\Python\seged.jpg", kep)
kepEredeti = mpimg.imread(r"C:\Python\seged.jpg") 

kep = cv2.resize(kep, (600,400))                                                        
kepSzurke = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)       
kepSzurke = cv2.dilate(kepSzurke, kernel, iterations = 1)
kepSzurke = cv2.bilateralFilter(kepSzurke, 13, 15, 15)

eldetektalt = cv2.Canny(kepSzurke,30,200)                                               # canny eldetektalas (a parameterekkel a tresholdot min,maxot adjuk meg)
kontur = cv2.findContours(eldetektalt.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # a korvonalak megtalalasahoz a hatternek feketenek kell lennie, az el pdig feher ezert, kell a binarys forditas es a canny eldetektalas (kep, a kinyeresi logika, kozelites) )
kontur = imutils.grab_contours(kontur)
kontur = sorted(kontur, key = cv2.contourArea, reverse = True)[:10]

screenCnt = None

for k in kontur:
    
    peri = cv2.arcLength(k, True)
    kozelites = cv2.approxPolyDP(k, 0.018 * peri, True)
 
    if len(kozelites) == 4:
        screenCnt = kozelites
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(kep, [screenCnt], -1, (0, 255, 255), 3)

mask = np.zeros(kepSzurke.shape,np.uint8)
keretezett_rendszam = cv2.drawContours(mask,[screenCnt],0 ,255, -1,)
keretezett_rendszam = cv2.bitwise_and(kep,kep,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
rendszam = kepSzurke[topx:bottomx+1, topy:bottomy+1]

rendszam = cv2.resize(rendszam,(500,200))

cv2.imwrite(r"C:\Python\rendszam.jpg", rendszam)

rendszam = cv2.imread(r"C:\Python\rendszam.jpg") 
szurke_rendszam = cv2.cvtColor (rendszam, cv2.COLOR_BGR2GRAY)
szurke_rendszam = cv2.GaussianBlur(szurke_rendszam,(5,5),0)                       #a blur miatt pontosabb eredmenyt lehet kapni, az igaz, hogy minel magasabb a blur annal pontosabb, de elveszhetnek fontos infok
binary_rendszam = cv2.threshold(szurke_rendszam, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
binary_rendszam = cv2.morphologyEx(binary_rendszam, cv2.MORPH_OPEN, kernel2, iterations=1)
binary_rendszam = 255 - binary_rendszam

text = tess.image_to_string(binary_rendszam, lang ='eng', config=' --psm 6 ')

utolsoChar = len(text)
text2 = ""
for x in range (utolsoChar-2):
    text2 = text2 + text[x]

print("Rendszam:",text2)

plt.figure(1, figsize=(20, 8))

plt.subplot(131), plt.title("Eredeti"), plt.imshow(kepEredeti), plt.axis("off")
plt.subplot(132), plt.title("Szürke"), plt.imshow(kepSzurke, cmap="gray"), plt.axis("off")
plt.subplot(133), plt.title("Éldetektált"), plt.imshow(eldetektalt, cmap="gray"), plt.axis("off")

plt.figure(2, figsize=(16, 6))

plt.subplot(131), plt.title("Keretezett rendszám"), plt.imshow(kep), plt.axis("off")
plt.subplot(132), plt.title("Szürke rendszám"), plt.imshow(szurke_rendszam, cmap="gray"), plt.axis("off")
plt.subplot(133), plt.title(text2), plt.imshow(binary_rendszam, cmap="gray"), plt.axis("off")

plt.show()
