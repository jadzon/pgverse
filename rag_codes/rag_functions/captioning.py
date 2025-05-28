import os
import gc
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class ImageCaptioner:
    def __init__(self, preload_models=True, caption_model_name="Salesforce/blip2-opt-2.7b", 
                 translation_model_name="facebook/m2m100_1.2B", use_gpu=True):
        """
        Initialize the ImageCaptioner class.
        
        Args:
            preload_models (bool): Whether to load models at initialization or on first use
            caption_model_name (str): Name of the BLIP-2 model to use
            translation_model_name (str): Name of the M2M100 model for translation
            use_gpu (bool): Whether to use GPU if available
        """
        # Configure GPU memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Check CUDA availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Model configuration
        self.caption_model_name = caption_model_name
        self.translation_model_name = translation_model_name
        
        # Initialize models as None
        self.processor = None
        self.caption_model = None
        self.m2m_tokenizer = None
        self.m2m_model = None
        
        # Load models if requested
        if preload_models:
            self._load_caption_model()
            self._load_translation_model()
    
    def _load_caption_model(self):
        """Load BLIP-2 captioning model"""
        if self.caption_model is None:
            try:
                # Clear CUDA memory before loading
                if self.use_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                self.processor = Blip2Processor.from_pretrained(self.caption_model_name)
                self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    self.caption_model_name,
                    device_map="auto" if self.use_gpu else None,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    offload_folder="offload"
                )
            except Exception as e:
                print(f"Error loading caption model: {e}")
                if self.use_gpu and "CUDA out of memory" in str(e):
                    print("Falling back to CPU for captioning model")
                    self.use_gpu = False
                    self.device = "cpu"
                    self._load_caption_model()  # Try again with CPU
    
    def _load_translation_model(self):
        """Load M2M100 translation model"""
        if self.m2m_model is None:
            try:
                # Clear CUDA memory before loading
                if self.use_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                self.m2m_tokenizer = M2M100Tokenizer.from_pretrained(self.translation_model_name)
                self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(
                    self.translation_model_name,
                    device_map="auto" if self.use_gpu else None,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    offload_folder="offload"
                )
            except Exception as e:
                print(f"Error loading translation model: {e}")
                if self.use_gpu and "CUDA out of memory" in str(e):
                    print("Falling back to CPU for translation model")
                    self.use_gpu = False
                    self.device = "cpu"
                    self._load_translation_model()  # Try again with CPU
    
    def describe_image(self, image_path):
        """
        Generate a Polish description of the image at the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Polish description of the image or error message
        """
        try:
            # Load models if not already loaded
            if self.caption_model is None:
                self._load_caption_model()
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Generate caption
            prompt = "Describe this image in detail:"
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    early_stopping=True
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()
            
            # Free up memory after captioning
            if self.use_gpu:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Translate to Polish
            return self.translate_to_polish(caption)
            
        except Exception as e:
            return f"Error describing image: {str(e)}"
    
    def translate_to_polish(self, text):
        """
        Translate text from English to Polish.
        
        Args:
            text (str): English text to translate
            
        Returns:
            str: Polish translation or error message
        """
        try:
            # Load translation model if not already loaded
            if self.m2m_model is None:
                self._load_translation_model()
            
            # Set languages
            src_lang = "en"
            tgt_lang = "pl"
            self.m2m_tokenizer.src_lang = src_lang
            
            # Translate text
            translation_inputs = self.m2m_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                translated_ids = self.m2m_model.generate(
                    **translation_inputs,
                    forced_bos_token_id=self.m2m_tokenizer.get_lang_id(tgt_lang),
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )
            translated_text = self.m2m_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            return translated_text
            
        except Exception as e:
            return f"Error translating text: {str(e)}"
    
    def cleanup(self):
        """Free up resources when done using the captioner"""
        if self.caption_model is not None:
            del self.caption_model
            self.caption_model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.m2m_model is not None:
            del self.m2m_model
            self.m2m_model = None
            
        if self.m2m_tokenizer is not None:
            del self.m2m_tokenizer
            self.m2m_tokenizer = None
        
        if self.use_gpu:
            torch.cuda.empty_cache()
            gc.collect()


# Example usage
if __name__ == "__main__":
    captioner = ImageCaptioner(preload_models=False)  # Lazy loading
    description = captioner.describe_image("input/image.jpg")
    print("Opis obrazu:", description)
    captioner.cleanup()