import re
import difflib

def measure_similarity_edit_distance(ref_text, test_text):
    """
    1) Usuwa wszystkie białe znaki (spacje, entery, tabulatory),
    2) Dopasowuje ciągi za pomocą difflib.SequenceMatcher,
    3) Oblicza koszt wstawień, usunięć i podmian,
    4) Zwraca (w procentach) 1 - (koszt / długość_ref_clean) * 100.
    """
    
    ref_clean = re.sub(r"\s+", "", ref_text)
    test_clean = re.sub(r"\s+", "", test_text)
    
   
    matcher = difflib.SequenceMatcher(None, ref_clean, test_clean)
    
    
    cost = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            cost += (j2 - j1)
        elif tag == 'delete':
            cost += (i2 - i1)
        elif tag == 'replace':
            cost += max((i2 - i1), (j2 - j1))
        
    
   
    ref_len = len(ref_clean)
    if ref_len == 0:
       
        return 0.0 if len(test_clean) > 0 else 100.0
    
    ratio = 1 - cost / ref_len
    
    ratio = max(ratio, 0.0)
    return ratio * 100


if __name__ == "__main__":
    ref  = "Ania ma psa i kota i szynszyle i chomika i parrota i wgl ma se duzo zwierzat"   
    test = "Ania ma psa!!!!!!!!!!!!!! i kota i szynszyle i chomika i parrota i wgl ma se duzo zwierzat"  
    
    percent = measure_similarity_edit_distance(ref, test)
    print(f"Zgodność: {percent:.2f}%")
    
