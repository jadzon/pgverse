import os
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from text_similarity import measure_lcs_similarity_percent, read_text_file


pytesseract.pytesseract.tesseract_cmd = 'D:\\nauka\\tesseract z instalatora\\tesseract.exe'

PSM_MODES = [0, 1,2, 3, 4,5, 6, 11, 12, 13]  
OEM_MODES = [0, 1, 2, 3]  


FILE_PAIRS = [
    {
        "pdf": "D:\\nauka\\baza\\k1.pdf", 
        "reference": "D:\\nauka\\baza\\k1.txt",
        "name": "k1"
    },
    {
        "pdf": "D:\\nauka\\baza\\k2.pdf", 
        "reference": "D:\\nauka\\baza\\k2.txt",
        "name": "k2"
    },
    {
        "pdf": "D:\\nauka\\baza\\k3.pdf", 
        "reference": "D:\\nauka\\baza\\k3.txt",
        "name": "k3"
    }
]

def process_pdf_with_params(pdf_path, psm, oem):
    """Process a PDF with specific PSM and OEM parameters."""
    try:
        
        images = convert_from_path(pdf_path)
        
        full_text = ""
        for i, img in enumerate(images):
            config = f"--psm {psm} --oem {oem}"
            text = pytesseract.image_to_string(img, lang="pol", config=config)
            full_text += f"=== Strona {i+1} ===\n{text}\n\n"
            
        return full_text
    except Exception as e:
        print(f"Error processing {pdf_path} with PSM={psm}, OEM={oem}: {e}")
        return ""

def main():
    
    results = []
    
    
    output_dir = "tesseract_results"
    os.makedirs(output_dir, exist_ok=True)
    
    
    total_combinations = len(FILE_PAIRS) * len(PSM_MODES) * len(OEM_MODES)
    current_combination = 0
    
    for file_info in FILE_PAIRS:
        pdf_path = file_info["pdf"]
        ref_path = file_info["reference"]
        file_name = file_info["name"]
        
        
        ref_text = read_text_file(ref_path)
        if ref_text is None:
            print(f"Could not read reference file: {ref_path}")
            continue
        
        for psm in PSM_MODES:
            for oem in OEM_MODES:
                current_combination += 1
                print(f"Processing combination {current_combination}/{total_combinations}: "
                      f"File: {file_name}, PSM={psm}, OEM={oem}")
                
                
                output_file = os.path.join(output_dir, f"{file_name}_psm{psm}_oem{oem}.txt")
                
                
                ocr_text = process_pdf_with_params(pdf_path, psm, oem)
                
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                
                
                similarity = measure_lcs_similarity_percent(ref_text, ocr_text)
                
                
                results.append({
                    "file": file_name,
                    "psm": psm,
                    "oem": oem,
                    "similarity": similarity,
                    "output_file": output_file
                })
    
    
    df_results = pd.DataFrame(results)
    
   
    results_csv = os.path.join(output_dir, "tesseract_results.csv")
    df_results.to_csv(results_csv, index=False)
    
    
    best_results = df_results.loc[df_results.groupby('file')['similarity'].idxmax()]
    
    
    report_path = os.path.join(output_dir, "tesseract_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tesseract OCR Parameter Optimization Results\n\n")
        f.write("## Best Parameters for Each File:\n\n")
        
        for _, row in best_results.iterrows():
            f.write(f"File: {row['file']}\n")
            f.write(f"- Best PSM: {row['psm']}\n")
            f.write(f"- Best OEM: {row['oem']}\n")
            f.write(f"- Similarity: {row['similarity']:.2f}%\n")
            f.write(f"- Output file: {row['output_file']}\n\n")
            
        f.write("\n## All Results (sorted by similarity):\n\n")
        
        sorted_results = df_results.sort_values(['file', 'similarity'], ascending=[True, False])
        
        last_file = None
        for _, row in sorted_results.iterrows():
            if last_file != row['file']:
                f.write(f"\nFile: {row['file']}\n")
                last_file = row['file']
                
            f.write(f"- PSM={row['psm']}, OEM={row['oem']}: {row['similarity']:.2f}%\n")
    
    print(f"\nOptimization complete!")
    print(f"Results saved to {results_csv}")
    print(f"Summary report saved to {report_path}")
    print("\nBest parameter combinations:")
    for _, row in best_results.iterrows():
        print(f"File {row['file']}: PSM={row['psm']}, OEM={row['oem']}, Similarity={row['similarity']:.2f}%")

if __name__ == "__main__":
    main()