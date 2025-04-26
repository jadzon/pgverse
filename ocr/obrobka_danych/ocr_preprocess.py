import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image

def negatyw(obraz):
    return cv2.bitwise_not(obraz)


def do_szarosci(obraz):
    return cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)


def progowanie(szarosc, prog=210, max_wartosc=230):
    _, binarny = cv2.threshold(szarosc, prog, max_wartosc, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarny


def usun_szum(obraz):
    rdzen = np.ones((1, 1), np.uint8)
    oczyszczony = cv2.dilate(obraz, rdzen, iterations=1)
    oczyszczony = cv2.erode(oczyszczony, rdzen, iterations=1)
    oczyszczony = cv2.morphologyEx(oczyszczony, cv2.MORPH_CLOSE, rdzen)
    oczyszczony = cv2.medianBlur(oczyszczony, 1)
    return oczyszczony


def zwieksz_kontrast(obraz):
    odwrocony = cv2.bitwise_not(obraz)
    rdzen = np.ones((1, 1), np.uint8)
    pogrubiony = cv2.dilate(odwrocony, rdzen, iterations=1)
    return cv2.bitwise_not(pogrubiony)


def zmniejsz_kontrast(obraz):
    odwrocony = cv2.bitwise_not(obraz)
    rdzen = np.ones((1, 1), np.uint8)
    pocieniony = cv2.erode(odwrocony, rdzen, iterations=1)
    return cv2.bitwise_not(pocieniony)


def kat_pochylenia(obraz):
    kopia = obraz.copy()
    szary = cv2.cvtColor(kopia, cv2.COLOR_BGR2GRAY)
    rozmyty = cv2.GaussianBlur(szary, (9, 9), 0)
    _, prog = cv2.threshold(rozmyty, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rdzen = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilacja = cv2.dilate(prog, rdzen, iterations=2)

    kontury, _ = cv2.findContours(dilacja, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    najwiekszy = max(kontury, key=cv2.contourArea)
    prostokat = cv2.minAreaRect(najwiekszy)
    kat = prostokat[-1]
    if kat < -45:
        kat = 90 + kat
    return -1.0 * kat


def obroc_obraz(obraz, kat):
    wysokosc, szerokosc = obraz.shape[:2]
    srodek = (szerokosc // 2, wysokosc // 2)
    macierz = cv2.getRotationMatrix2D(srodek, kat, 1.0)
    return cv2.warpAffine(obraz, macierz, (szerokosc, wysokosc), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def wyrownaj(obraz):
    kat = kat_pochylenia(obraz)
    return obroc_obraz(obraz, kat)


def przytnij_marginesy(obraz):
    kontury, _ = cv2.findContours(obraz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obszar = sorted(kontury, key=lambda x: cv2.contourArea(x))[-1]
    x, y, w, h = cv2.boundingRect(obszar)
    return obraz[y:y + h, x:x + w]


def dodaj_margines(obraz, margines=150):
    kolor = [255, 255, 255]
    return cv2.copyMakeBorder(obraz, margines, margines, margines, margines,
                              cv2.BORDER_CONSTANT, value=kolor)


pdf_wejsciowy = "ep.pdf"
strony = convert_from_path(pdf_wejsciowy)
strony_do_pdf = []

for i, strona in enumerate(strony):
    obraz_cv = cv2.cvtColor(np.array(strona), cv2.COLOR_RGB2BGR)

    szary = do_szarosci(obraz_cv)
    prog = progowanie(szary)
    bez_szumu = usun_szum(prog)
    wyrownany = wyrownaj(obraz_cv)
    przyciety = przytnij_marginesy(bez_szumu)
    finalny = dodaj_margines(przyciety)

    finalny_rgb = cv2.cvtColor(finalny, cv2.COLOR_BGR2RGB)
    finalny_pil = Image.fromarray(finalny_rgb)
    strony_do_pdf.append(finalny_pil)

if strony_do_pdf:
    strony_do_pdf[0].save("output.pdf", save_all=True, append_images=strony_do_pdf[1:])

print("Nowy PDF po preprocessingu zapisany jako output.pdf")
