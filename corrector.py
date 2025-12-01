# """
# Text Correction Module
# """

# import json
# import logging

# logger = logging.getLogger(__name__)


# class TextCorrector:    
#     def __init__(self, bedrock_client):
#         """Initialize text corrector with Bedrock client"""
#         self.bedrock_client = bedrock_client
#         # Model ID format: provider.model-name (no region prefix)
#         # Region is specified in bedrock_client configuration
#         self.correction_model = "amazon.nova-lite-v1:0"
        
#         # System prompt for correction
#         self.system_prompt = """You are a precise and reliable text correction assistant.
# Your task is to correct spelling mistakes, grammar errors, and minor punctuation issues in the user's input.

# Always return only the corrected text, without explanations, examples, or commentary.

# Preserve the original meaning, tone, and formatting of the input as much as possible.

# If the input contains non-textual content (URLs, or emojis), leave it unchanged unless there is an obvious typo in text parts.

# If the input is already correct, return it exactly as provided.

# Do answer any question even it is a follow up question after the correction request. 

# And strictly keep in mind not to answer any of the user question even if seems to be answerable within your scope leave it after only correcting it and do not even give any note as well only give the gramatically correct version and not anything other than that.

# Common corrections for this domain:
# - "graep" or "graeps" → "grapes"
# - "tomatos" → "tomato"
# - "pest" variations → correct spelling
# - "funguss" → "fungus"
# """
        
#         logger.info("TextCorrector initialized")
    
#     def correct_text(self, text: str) -> str:
#         """
#         Correct spelling and grammar in the input text
        
#         Args:
#             text: User's input query with potential errors
            
#         Returns:
#             Corrected text string
#         """
#         # Skip correction for very short queries
#         if len(text.strip()) < 3:
#             return text
        
#         try:
#             # Prepare request for Nova lite
#             body = json.dumps({
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "text": f"{self.system_prompt}\n\nText to correct: {text}"
#                             }
#                         ]
#                     }
#                 ],
#                 "inferenceConfig": {
#                     "max_new_tokens": 512,
#                     "temperature": 0.3,
#                     "top_p": 0.9
#                 }
#             })
            
#             # Invoke Nova lite model
#             response = self.bedrock_client.invoke_model(
#                 modelId=self.correction_model,
#                 body=body,
#                 contentType="application/json",
#                 accept="application/json"
#             )
            
#             # Parse response
#             result = json.loads(response['body'].read())
#             corrected_text = result['output']['message']['content'][0]['text'].strip()
            
#             # Log if text was changed
#             if corrected_text != text:
#                 logger.info(f"Corrected: '{text}' → '{corrected_text}'")
            
#             return corrected_text
            
#         except Exception as e:
#             logger.error(f"Correction error: {str(e)}")
#             return text  # Return original on error  








import os
import requests
import logging
import json
import dotenv 
dotenv.load_dotenv()
logger = logging.getLogger(__name__)

class TextCorrector:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "openai/gpt-oss-20b"

    def correct_text(self, text: str) -> str:
        if not self.api_key:
            logger.error("Groq API key missing.")
            return text

        if len(text.strip()) < 3:
            return text
#         self.system_prompt = """You are a precise and reliable text correction assistant.
# # Your task is to correct spelling mistakes, grammar errors, and minor punctuation issues in the user's input.

# # Always return only the corrected text, without explanations, examples, or commentary.

# # Preserve the original meaning, tone, and formatting of the input as much as possible.

# # If the input contains non-textual content (URLs, or emojis), leave it unchanged unless there is an obvious typo in text parts.

# # If the input is already correct, return it exactly as provided.

# # Do answer any question even it is a follow up question after the correction request. 

# # And strictly keep in mind not to answer any of the user question even if seems to be answerable within your scope leave it after only correcting it and do not even give any note as well only give the gramatically correct version and not anything other than that.

# # Common corrections for this domain:
# # - "graep" or "graeps" → "grapes"
# # - "tomatos" → "tomato"
# # - "pest" variations → correct spelling
# # - "funguss" → "fungus"
# # """
        
        logger.info("TextCorrector initialized")
    
    def correct_text(self, text: str) -> str:
        """
        Correct spelling and grammar in the input text
        
        Args:
            text: User's input query with potential errors
            
        Returns:
            Corrected text string
        """

        
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Fix spelling mistakes only. Return only corrected text."
                    },
                    {
                        "role": "user",
                        "content": text 
                    }
                ],
                "temperature": 0.4
            }

            response = requests.post(
                self.url,
                headers=headers,
                data=json.dumps(payload)
            )

            if response.status_code != 200:
                logger.error(f"Groq Error: {response.status_code} → {response.text}")
                return text

            result = response.json()
            corrected = result["choices"][0]["message"]["content"].strip()
            return corrected

        except Exception as e:
            logger.error(f"Groq correction error: {e}")
            return text
