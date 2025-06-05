import sys
import os
from pathlib import Path

from rag_functions.metadata_caption import ImageTextProcessor


def main():
    """
    Główna funkcja do przetwarzania k6.txt na k6_base64.txt i k6_chunks.txt
    """
    print("=== KONWERTER K6.TXT NA BASE64 I CHUNKS ===")
    
    # Parametry
    input_file = "k6.txt"
    base64_output_file = "k6_base64.txt"
    chunks_output_file = "k6_chunks.txt"
    
    # Sprawdź czy plik k6.txt istnieje
    if not os.path.exists(input_file):
        print(f"❌ Błąd: Plik {input_file} nie istnieje!")
        print("Upewnij się, że plik k6.txt znajduje się w tym samym folderze co skrypt.")
        
        # Pokaż dostępne pliki txt
        txt_files = list(Path(".").glob("*.txt"))
        if txt_files:
            print("Dostępne pliki .txt:")
            for txt_file in txt_files:
                print(f"  - {txt_file.name}")
        return
    
    # Sprawdź folder figury
    figury_path = Path("figury")
    if figury_path.exists():
        image_files = list(figury_path.glob("*.png")) + list(figury_path.glob("*.jpg")) + list(figury_path.glob("*.jpeg"))
        print(f"✅ Znaleziono folder figury z {len(image_files)} obrazami")
        
        # Pokaż kilka przykładów
        if image_files:
            print("Przykładowe obrazy:")
            for img in image_files[:3]:
                print(f"  - {img.name}")
            if len(image_files) > 3:
                print(f"  ... i jeszcze {len(image_files) - 3} obrazów")
    else:
        print(f"⚠️  Ostrzeżenie: Folder figury nie istnieje w {figury_path.absolute()}")
    
    print(f"\n🔄 Rozpoczynam przetwarzanie {input_file}...")
    
    try:
        # Utwórz procesor
        processor = ImageTextProcessor(max_tokens=150)
        
        # === PRZETWARZANIE NA BASE64 ===
        print("\n📦 Tworzenie pliku z obrazami jako base64...")
        base64_result = processor.process_file_to_txt_with_base64(input_file, base64_output_file)
        
        if base64_result:
            print(f"✅ Sukces! Utworzono plik base64: {base64_result}")
            
            # Pokaż statystyki base64
            show_base64_stats(base64_result)
        else:
            print("❌ Błąd podczas tworzenia pliku base64!")
        
        # === PRZETWARZANIE NA CHUNKS ONLY ===
        print("\n📝 Tworzenie pliku z samymi chunkami tekstowymi...")
        chunks_result = processor.process_file_to_txt_chunks_only(input_file, chunks_output_file)
        
        if chunks_result:
            print(f"✅ Sukces! Utworzono plik chunks: {chunks_result}")
            
            # Pokaż statystyki chunks
            show_chunks_stats(chunks_result)
        else:
            print("❌ Błąd podczas tworzenia pliku chunks!")
            
        # === PODSUMOWANIE ===
        print("\n" + "="*60)
        print("📋 PODSUMOWANIE:")
        if base64_result:
            print(f"  📦 Plik base64: {base64_result}")
        if chunks_result:
            print(f"  📝 Plik chunks: {chunks_result}")
        print("✅ Przetwarzanie zakończone!")
            
    except Exception as e:
        print(f"❌ Wystąpił błąd: {e}")
        import traceback
        print("Szczegóły błędu:")
        traceback.print_exc()


def show_base64_stats(file_path):
    """
    Pokazuje statystyki pliku base64
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        image_lines = [line for line in lines if line.startswith('<image/')]
        text_lines = [line for line in lines if line.strip() and not line.startswith('<image/')]
        error_lines = [line for line in lines if 'ERROR_CONVERTING' in line]
        
        print(f"    📊 Statystyki pliku {file_path}:")
        print(f"      📝 Chunki tekstowe: {len(text_lines)}")
        print(f"      🖼️  Obrazy (base64): {len(image_lines) - len(error_lines)}")
        if error_lines:
            print(f"      ❌ Błędy konwersji: {len(error_lines)}")
        print(f"      📄 Łączna liczba linii: {len([line for line in lines if line.strip()])}")
        
        # Pokaż rozmiar pliku
        file_size = os.path.getsize(file_path)
        if file_size > 1024*1024:
            print(f"      💾 Rozmiar pliku: {file_size/(1024*1024):.1f} MB")
        elif file_size > 1024:
            print(f"      💾 Rozmiar pliku: {file_size/1024:.1f} KB")
        else:
            print(f"      💾 Rozmiar pliku: {file_size} bajtów")
        
    except Exception as e:
        print(f"    ❌ Nie można odczytać statystyk: {e}")


def show_chunks_stats(file_path):
    """
    Pokazuje statystyki pliku chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Podziel na chunki (oddzielone pustymi liniami)
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        print(f"    📊 Statystyki pliku {file_path}:")
        print(f"      📝 Liczba chunków: {len(chunks)}")
        
        if chunks:
            # Średnia długość chunku
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            print(f"      📏 Średnia długość chunku: {avg_length:.0f} znaków")
            
            # Najkrótszy i najdłuższy chunk
            min_length = min(len(chunk) for chunk in chunks)
            max_length = max(len(chunk) for chunk in chunks)
            print(f"      📐 Najkrótszy chunk: {min_length} znaków")
            print(f"      📏 Najdłuższy chunk: {max_length} znaków")
            
            # Pokaż pierwszy chunk jako przykład
            first_chunk = chunks[0]
            preview = first_chunk[:100] + "..." if len(first_chunk) > 100 else first_chunk
            print(f"      👁️  Przykład pierwszego chunku: '{preview}'")
        
        # Rozmiar pliku
        file_size = os.path.getsize(file_path)
        if file_size > 1024:
            print(f"      💾 Rozmiar pliku: {file_size/1024:.1f} KB")
        else:
            print(f"      💾 Rozmiar pliku: {file_size} bajtów")
        
    except Exception as e:
        print(f"    ❌ Nie można odczytać statystyk: {e}")


def test_chunks_only_function():
    """
    Testuje funkcję create_output_txt_chunks_only bezpośrednio
    """
    print("\n🧪 TESTOWANIE FUNKCJI CHUNKS-ONLY...")
    
    try:
        processor = ImageTextProcessor(max_tokens=50)
        
        # Przykładowy tekst z obrazami
        test_content = """
        To jest pierwszy chunk tekstu przed obrazem.
        Zawiera informacje o procesorze.
        
        <image/figury/cpu.png>
        
        To jest drugi chunk tekstu po obrazie.
        Opisuje pamięć RAM i jej parametry.
        
        <image/figury/ram.jpg>
        
        To jest ostatni chunk tekstu.
        Zawiera podsumowanie specyfikacji.
        """
        
        # Przetwórz tekst
        texts = processor.process_text(test_content)
        print(f"    📋 Znaleziono {len(texts)} elementów")
        
        # Policz typy elementów
        text_count = sum(1 for t in texts if processor.get_element_type(t) == 'text')
        image_count = sum(1 for t in texts if processor.get_element_type(t) == 'image')
        print(f"    📝 Chunki tekstowe: {text_count}")
        print(f"    🖼️  Obrazy: {image_count}")
        
        # Utwórz plik chunks-only
        chunks_file = processor.create_output_txt_chunks_only(texts, "test_chunks.txt")
        
        if chunks_file:
            print(f"    ✅ Utworzono testowy plik: {chunks_file}")
            
            # Sprawdź zawartość
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_content = f.read()
            
            chunks = [chunk.strip() for chunk in chunks_content.split('\n\n') if chunk.strip()]
            print(f"    📊 Plik zawiera {len(chunks)} chunków")
            
            # Pokaż chunki
            for i, chunk in enumerate(chunks, 1):
                preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
                print(f"    {i}. '{preview}'")
                
        else:
            print("    ❌ Błąd tworzenia pliku chunks")
            
    except Exception as e:
        print(f"    ❌ Błąd testu: {e}")
        import traceback
        traceback.print_exc()


def test_import():
    """
    Testuje czy import działa poprawnie
    """
    try:
        from rag_functions.metadata_caption import ImageTextProcessor
        print("✅ Import metadata_caption.py działa poprawnie")
        
        processor = ImageTextProcessor()
        print("✅ Utworzenie ImageTextProcessor działa poprawnie")
        
        # Test funkcji chunks-only
        if hasattr(processor, 'create_output_txt_chunks_only'):
            print("✅ Funkcja create_output_txt_chunks_only dostępna")
        else:
            print("❌ Funkcja create_output_txt_chunks_only NIE DOSTĘPNA")
            
        if hasattr(processor, 'process_file_to_txt_chunks_only'):
            print("✅ Funkcja process_file_to_txt_chunks_only dostępna")
        else:
            print("❌ Funkcja process_file_to_txt_chunks_only NIE DOSTĘPNA")
        
        return True
    except Exception as e:
        print(f"❌ Błąd importu: {e}")
        return False


if __name__ == "__main__":
    print("Testowanie importu...")
    if test_import():
        # Test funkcji chunks-only
        test_chunks_only_function()
        
        print()
        main()
    else:
        print("\nNie można kontynuować - błąd importu.")
        print("Sprawdź czy:")
        print("1. Plik metadata_caption.py znajduje się w folderze rag_functions/")
        print("2. Folder rag_functions/ zawiera plik __init__.py")
        print("3. Wszystkie zależności są zainstalowane (PIL, sklearn, etc.)")