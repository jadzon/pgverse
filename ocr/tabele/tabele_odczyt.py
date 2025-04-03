import os
from img2table.document import Image
from img2table.ocr import TesseractOCR


input_folder = "results/tabele"
output_folder = "wyniki_csv"


os.makedirs(output_folder, exist_ok=True)


ocr = TesseractOCR(lang="pol")


for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        print( {filename})


        img = Image(image_path)

        # Wydobycie tabel
        extracted_tables = img.extract_tables(ocr=ocr)


        if extracted_tables:
            for i, table in enumerate(extracted_tables, start=1):
                output_name = f"{os.path.splitext(filename)[0]}_tabela{i}.csv"
                output_path = os.path.join(output_folder, output_name)
                table.df.to_csv(output_path, index=False)
                print(f" Zapisano: {output_path}")
        else:
            print(" Nie wykryto tabel.")
