# Schematics

Ten moduł jest odpowiedzialny za analizę i przetwarzanie schematów elektronicznych i automatyki. Proces obejmuje etapy od wstępnego przetwarzania obrazu, przez detekcję poszczególnych komponentów elektronicznych, aż po wykrywanie połączeń między nimi.

## Struktura projektu

Projekt jest podzielony na następujące moduły:

- **preprocessing/** - Moduł wstępnego przetwarzania obrazu
- **component_detection/** - Wykrywanie komponentów elektronicznych
- **net_detector/** - Wykrywanie połączeń między komponentami

## Opis modułów

### 1. Preprocessing

Moduł odpowiedzialny za wstępne przetwarzanie obrazu schematu:

- Filtracja szumów
- Normalizacja kontrastu
- itd

### 2. Component Detection

Moduł wykrywa następujące elementy elektroniczne na schemacie:

- Źródło napięcia zmiennego
- Bateria/źródło napięcia stałego
- Kondensator
- Źródło prądu
- Źródło napięcia stałego
- Źródło sterowane
- Dioda
- Symbol uziemienia
- Cewka
- Tranzystor mosfet
- Tranzystor bipolarny
- Rezystor
- Woltomierz
- Amperomierz

### 3. Net Detector

Moduł odpowiedzialny za:

- Wykrywanie połączeń między komponentami
- Zapisywanie wyników w odpowiedniej strukturze

## Jak używać

```python
# Przykładowe użycie modułu
TODO wgl wszystko todo jest lipa, bierzemy się do roboty :(
```
