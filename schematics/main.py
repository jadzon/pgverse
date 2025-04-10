from block_detector.block_detector import BlockDetector
from net_detector.line_detector import LineDetector
from text_extraction.text_extraction import TextExtractor
from preprocessing.png_proc import Preprocessor
import os



def main():
    BlockDetector = BlockDetector()
    LineDetector = LineDetector()
    TextExtractor = TextExtractor()
    Preprocessor = Preprocessor()

if __name__ == "__main__":
    main()