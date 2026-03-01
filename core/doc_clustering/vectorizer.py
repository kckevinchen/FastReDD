import os
import logging
from openai import OpenAI

# Try importing Google Generative AI (optional dependency for Gemini)
# We'll check availability at runtime as well for robustness
# Note: Uses the new google-genai package (not the legacy google-generativeai package)
# Install with: pip install google-genai
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # Set to None so we can check at runtime


def llm_embeddings(string, model="text-embedding-3-small", api_key=None):
    """
    Generate text embeddings using OpenAI or Google Gemini embedding models.
    
    This function returns a vector (list of float values) representing the text embedding.
    
    OpenAI models:
        text-embedding-3-small    $0.020 / 1M tokens
        text-embedding-3-large    $0.130 / 1M tokens
        text-embedding-ada-002    $0.100 / 1M tokens
    
    Google Gemini embedding models:
        gemini-embedding-001      (Recommended format - will be converted to models/embedding-001)
        models/embedding-001      (Direct Gemini API format)
        embedding-001             (Will be converted to models/embedding-001)
    
    Args:
        string: Input text to embed
        model: Embedding model name (must be an embedding model, not a text generation model)
        api_key: Optional API key. If None, will try to get from environment variable.
                 For Gemini: GOOGLE_API_KEY or GEMINI_API_KEY
                 For OpenAI: OPENAI_API_KEY
        
    Returns:
        List of float values representing the embedding vector
    """
    # Check if it's a Gemini model
    is_gemini = (
        model.startswith("models/embedding-") or 
        model.startswith("embedding-") or
        model.startswith("gemini-embedding-")
    )
    
    if is_gemini:
        # Try to import genai at runtime if not already available
        # This handles cases where the library is installed after module import
        # or if the module wasn't imported successfully initially
        # Note: The correct import is "from google import genai"
        global genai, GEMINI_AVAILABLE
        try:
            # Check if we need to import genai
            if not GEMINI_AVAILABLE or genai is None:
                # Re-import at runtime using the correct import path
                from google import genai
                GEMINI_AVAILABLE = True
        except ImportError as e:
            logging.error(f"[get_gemini_embeddings] "
                         f"Google Generative AI library not installed. "
                         f"Install it with: pip install google-genai. "
                         f"Original error: {e}")
            raise ImportError(
                f"Google Generative AI library not installed. "
                f"Install it with: pip install google-genai. "
                f"Original error: {e}"
            )
        
        # Get API key: use provided key, or try environment variables
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logging.error(f"[get_gemini_embeddings] "
                         f"API key not provided and GOOGLE_API_KEY/GEMINI_API_KEY environment variable not set. "
                         f"Please provide an API key or set it as an environment variable to use Gemini embeddings.")
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY/GEMINI_API_KEY environment variable not set. "
                "Please provide an API key or set it as an environment variable to use Gemini embeddings."
            )
        
        # Use genai.Client() and client.models.embed_content() (correct API approach)
        try:
            # Initialize client (API key from environment variable is used automatically)
            # Can also pass api_key explicitly: genai.Client(api_key=api_key)
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            
            # Use model name as-is (e.g., "gemini-embedding-001")
            result = client.models.embed_content(
                model=model,
                contents=string
            )
            
            # Extract embeddings from result.embeddings
            # Based on the API: result.embeddings is a list of embedding objects
            if hasattr(result, 'embeddings'):
                embeddings = result.embeddings
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    # Get the first embedding (there should only be one for single content)
                    embedding = embeddings[0]
                    # The embedding object should have a 'values' attribute containing the vector
                    if hasattr(embedding, 'values'):
                        return list(embedding.values)
                    # Or it might be directly a list/tuple of floats
                    elif isinstance(embedding, (list, tuple)):
                        return list(embedding)
                elif isinstance(embeddings, (list, tuple)):
                    # If embeddings is directly a list of floats
                    return list(embeddings)
            
            logging.error(f"[get_gemini_embeddings] "
                         f"Unexpected response format from Gemini API. "
                         f"Result type: {type(result)}, "
                         f"Has embeddings attr: {hasattr(result, 'embeddings')}, "
                         f"result.embeddings type: {type(result.embeddings) if hasattr(result, 'embeddings') else 'N/A'}")
            raise ValueError(f"Unexpected response format from Gemini API. "
                           f"Result type: {type(result)}, "
                           f"Has embeddings attr: {hasattr(result, 'embeddings')}, "
                           f"result.embeddings type: {type(result.embeddings) if hasattr(result, 'embeddings') else 'N/A'}")
            
        except Exception as e:
            # Fallback: Try OpenAI-compatible endpoint if Client API fails
            try:
                from openai import OpenAI
                client_openai = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai")
                
                # Normalize model name for OpenAI-compatible endpoint
                if model.startswith("gemini-embedding-"):
                    gemini_model_openai = f"models/{model.replace('gemini-', '')}"
                elif model.startswith("embedding-") and not model.startswith("models/"):
                    gemini_model_openai = f"models/{model}"
                elif not model.startswith("models/"):
                    gemini_model_openai = f"models/{model}"
                else:
                    gemini_model_openai = model
                
                response = client_openai.embeddings.create(
                    model=gemini_model_openai,
                    input=string
                )
                return list(response.data[0].embedding)
                
            except Exception as e2:
                logging.error(f"[get_gemini_embeddings] "
                             f"Failed to get embeddings from Gemini API. "
                             f"Client API error: {e}. "
                             f"OpenAI-compatible endpoint error: {e2}. "
                             f"Model: {model}. "
                             f"Please check that the model name is correct and GOOGLE_API_KEY is set.")
                raise ValueError(
                    f"Failed to get embeddings from Gemini API. "
                    f"Client API error: {e}. "
                    f"OpenAI-compatible endpoint error: {e2}. "
                    f"Model: {model}. "
                    f"Please check that the model name is correct and GOOGLE_API_KEY is set."
                ) from e2
    
    else:
        # OpenAI models
        # Use provided API key or get from environment variable
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        client_gpt = OpenAI(api_key=api_key) if api_key else OpenAI()
        embeddings = client_gpt.embeddings.create(
            input=string,
            model=model
        )
        return embeddings.data[0].embedding


class Vectorizer:
    def __init__(self):
        pass

    def fit_transform(self):
        logging.error(f"[{self.__class__.__name__}:fit_transform] Not implemented")
        raise NotImplementedError

    def transform(self):
        logging.error(f"[{self.__class__.__name__}:transform] Not implemented")
        raise NotImplementedError


class DocVectorizer(Vectorizer):
    def __init__(self, embedder=None):
        super().__init__()
        self.embedder = embedder if embedder else llm_embeddings

    def fit_transform(self, documents):
        return [self.embedder(document) for document in documents]
    