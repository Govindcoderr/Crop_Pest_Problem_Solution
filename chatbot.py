
"""
Pesticide Recommendation Chatbot - TF-IDF + Groq Intent (Corrected)
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

import dotenv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load .env
dotenv.load_dotenv()

from database import PesticideDatabase
from session_manager import SessionState
from corrector import TextCorrector  # should be Groq-based TextCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEmbedding:
    """
    Local TF-IDF embeddings for semantic search.
    Works well for small/medium knowledge bases and avoids heavy DL deps.
    """
    def __init__(self, documents: List[str]):
        # Ensure at least one doc to fit vectorizer safely
        self.documents = documents or [""]
        self.vectorizer = TfidfVectorizer().fit(self.documents)
        self.doc_vectors = self.vectorizer.transform(self.documents)

    def embed(self, text: str):
        return self.vectorizer.transform([text])

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if not self.documents or (len(self.documents) == 1 and self.documents[0] == ""):
            return []
        query_vec = self.embed(query)
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for i in top_indices:
            if similarities[i] > 0:
                results.append(self.documents[i])
        return results


class PesticideChatbot:
    """
    Main chatbot class with structured data + TF-IDF RAG fallback + Groq intent detection.
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base/pesticide_recommendations.md"):
        self.knowledge_base_path = knowledge_base_path

        # Groq API config (read from .env)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        # Model name - use the model you have access to; change if needed
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        # Initialize components
        self.database = PesticideDatabase(knowledge_base_path)
        self.text_corrector = TextCorrector()  # your corrector should read GROQ_API_KEY internally

        # Load knowledge base and build TF-IDF index
        self.documents = self._load_documents()
        self.embedding_model = SimpleEmbedding(self.documents)

        logger.info("✅ PesticideChatbot initialized successfully!")

    def _load_documents(self) -> List[str]:
        """Load markdown and extract table rows as documents for semantic search."""
        if not os.path.exists(self.knowledge_base_path):
            logger.warning("Knowledge base file not found!")
            return []

        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.strip().split('\n')
        table_lines = [l for l in lines if l.strip().startswith('|')]
        data_lines = table_lines[2:] if len(table_lines) >= 3 else []
        # Normalize whitespace
        data_lines = [l.strip() for l in data_lines if l.strip()]
        return data_lines

    # ---------------------- Semantic Search (TF-IDF) ----------------------
    def _semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """Return top-k relevant document lines from knowledge base using TF-IDF."""
        if not self.documents:
            return []
        try:
            return self.embedding_model.search(query, top_k)
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    # ---------------------- Intent Detection (Groq) ----------------------
    def detect_intent(self, query: str, session: SessionState) -> Dict[str, Any]:
        """
        Use Groq LLM to classify intent and extract entities.
        Returns a dict matching required structure.
        Falls back to rule-based detection on error.
        """
        # Prepare system prompt that forces JSON output with entities
        system_prompt = (
            "You are an intent and entity extraction assistant for a pesticide recommendation chatbot.\n"
            "Extract intent_type and entities and ALWAYS return a single valid JSON object only.\n"
            "JSON structure must be exactly:\n"
            "{\n"
            '  "intent_type": "provide_info | uncertainty | denial | confirmation | question | greeting | farewell | off_topic | unclear | list_pest_problem | pest_inquiry",\n'
            '  "entities": { "crop": null_or_string, "pest_name": null_or_string, "application_type": null_or_string },\n'
            '  "uncertain_about": null_or_string,\n'
            '  "reasoning": "brief explanation",\n'
            '  "confidence": "high | medium | low"\n'
            "}\n"
            "If a field is not applicable use null. Do not include any extra text, explanation or markdown—only the JSON."
        )

        user_context = (
            f"Conversation History: {session.get_history_for_llm(max_messages=4) or 'None'}\n"
            f"Last Bot Message: {session.get_last_bot_message() or 'None'}\n"
            f"Session crop: {session.crop or 'Unknown'}\n"
            f"Session pest: {session.pest_name or 'Unknown'}\n"
            f"\nUser's new message: \"{query}\"\n"
            "Classify intent and extract entities based on the message and context."
        )

        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set. Falling back to rule-based intent detection.")
            return self._intent_fallback(query)

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }

        try:
            resp = requests.post(self.groq_url, headers=headers, json=payload, timeout=15)
            if resp.status_code != 200:
                logger.error(f"Groq intent error: {resp.status_code} → {resp.text}")
                return self._intent_fallback(query)

            body = resp.json()
            # Groq chat completions follow OpenAI-like structure: choices[0].message.content
            text_output = body.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not text_output:
                logger.error("Empty Groq response; using fallback")
                return self._intent_fallback(query)

            # Extract JSON substring if assistant wrapped it with code fences or text
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if not json_match:
                logger.error(f"Could not find JSON in Groq output: {text_output!r}")
                return self._intent_fallback(query)

            parsed = json.loads(json_match.group(0))

            # Defensive sanitization
            if 'entities' not in parsed or parsed['entities'] is None:
                parsed['entities'] = {"crop": None, "pest_name": None, "application_type": None}
            else:
                # ensure keys exist
                for k in ("crop", "pest_name", "application_type"):
                    parsed['entities'].setdefault(k, None)

            # normalize uncertain_about / confidence
            parsed.setdefault('uncertain_about', None)
            parsed.setdefault('reasoning', '')
            parsed.setdefault('confidence', 'medium')

            return parsed

        except Exception as e:
            logger.error(f"Groq intent detection failed → fallback. Error: {e}")
            return self._intent_fallback(query)

    def _intent_fallback(self, query: str) -> Dict[str, Any]:
        """Simple rule-based fallback intent detection."""
        q = query.lower()
        intent = {"intent_type": "unclear", "entities": {"crop": None, "pest_name": None, "application_type": None}, "uncertain_about": None, "reasoning": "", "confidence": "low"}

        if any(w in q for w in ["hello", "hi", "hey"]):
            intent["intent_type"] = "greeting"
            intent["confidence"] = "high"
            return intent

        if any(w in q for w in ["bye", "thanks", "thank you", "ok thanks"]):
            intent["intent_type"] = "farewell"
            intent["confidence"] = "high"
            return intent

        if "list" in q and "pest" in q:
            # try to extract crop if present
            # Very simple extraction using database crop list
            crops = [c.lower() for c in self.database.get_most_common_crops(limit=50)]
            found_crop = None
            for c in crops:
                if c in q:
                    found_crop = c
                    break
            intent["intent_type"] = "list_pest_problem"
            intent["entities"]["crop"] = found_crop
            intent["confidence"] = "medium"
            return intent

        if any(w in q for w in ["don't know", "dont know", "not sure", "not certain"]):
            intent["intent_type"] = "uncertainty"
            intent["uncertain_about"] = "pest_name"
            intent["confidence"] = "medium"
            return intent

        # If contains crop like "wheat", mark provide_info
        crops = [c.lower() for c in self.database.get_most_common_crops(limit=50)]
        for c in crops:
            if c in q:
                intent["intent_type"] = "provide_info"
                intent["entities"]["crop"] = c
                intent["confidence"] = "medium"
                # try to find pest name words (very naive)
                # If user contains words like 'fungus', 'aphid', set pest_name
                for pest_word in ["aphid", "whitefly", "fungus", "borer", "mildew", "blight", "stem borer", "stem-borer"]:
                    if pest_word in q:
                        intent["entities"]["pest_name"] = pest_word
                        break
                return intent

        # question detection
        if "?" in q or any(w in q for w in ["how", "what", "which", "when", "where"]):
            intent["intent_type"] = "question"
            intent["confidence"] = "medium"
            return intent

        # default
        intent["intent_type"] = "provide_info"
        intent["confidence"] = "low"
        return intent

    # ---------------------- Intent Validation ----------------------
    def _validate_intent(self, intent: Any) -> Dict[str, Any]:
        """Ensure intent is a well-formed dict with required keys."""
        if not isinstance(intent, dict):
            return {"intent_type": "unclear", "entities": {"crop": None, "pest_name": None, "application_type": None}, "uncertain_about": None, "reasoning": "Invalid intent format", "confidence": "low"}
        intent.setdefault('intent_type', 'unclear')
        entities = intent.get('entities') or {}
        if not isinstance(entities, dict):
            entities = {}
        # Ensure entity keys are present (and normalize empty strings -> None)
        for k in ("crop", "pest_name", "application_type"):
            val = entities.get(k) if k in entities else None
            if isinstance(val, str) and val.strip() == "":
                val = None
            entities[k] = val
        intent['entities'] = entities
        intent.setdefault('uncertain_about', None)
        intent.setdefault('reasoning', '')
        intent.setdefault('confidence', 'low')
        return intent

    # ---------------------- Response Formatting ----------------------
    def format_solution_response(self, solutions: List[Dict], crop: str, pest: Optional[str] = None) -> str:
        """Format solutions as structured markdown text."""
        if not solutions:
            return "No solutions found for the specified criteria."

        if not pest:
            pests_dict = {}
            for sol in solutions:
                p = sol.get('pest_name') or 'Unknown'
                pests_dict.setdefault(p, []).append(sol)

            output = f"# 🌾 Solutions for {crop}\n\n"
            for pest_name, pest_solutions in pests_dict.items():
                output += f"## {pest_name}\n\n"
                for idx, sol in enumerate(pest_solutions, 1):
                    output += f"**Solution {idx}:**\n"
                    output += f"- **Product:** {sol.get('solution','N/A')}\n"
                    output += f"- **Method:** {sol.get('application','N/A')}\n"
                    output += f"- **Dosage:** {sol.get('dosage','N/A')}\n"
                    output += f"- **Waiting Period:** {sol.get('waiting_period','N/A')} days\n\n"
            return output

        output = f"# 🌾 {crop} - {pest}\n\n"
        for idx, sol in enumerate(solutions, 1):
            output += f"**Solution {idx}:**\n"
            output += f"- **Product:** {sol.get('solution','N/A')}\n"
            output += f"- **Method:** {sol.get('application','N/A')}\n"
            output += f"- **Dosage:** {sol.get('dosage','N/A')}\n"
            output += f"- **Waiting Period:** {sol.get('waiting_period','N/A')} days\n\n"
        return output

    # ---------------------- Query Handlers ----------------------
    def _handle_greeting(self) -> str:
        return ("Hello! I can help you with pesticide recommendations.\n"
                "Tell me which crop you are working with and what pest/disease you're facing.")

    def _handle_farewell(self) -> str:
        return "You're welcome — feel free to ask again if you need more help."

    def _handle_unclear(self) -> str:
        return "I couldn't understand that. Could you provide the crop and the pest/disease or describe the symptoms?"

    def _handle_off_topic(self) -> str:
        return "I specialize in pesticide recommendations. I cannot help with weather or market prices."

    def _handle_uncertainty(self, intent: Dict, session: SessionState) -> str:
        uncertain_about = intent.get('uncertain_about')
        if uncertain_about == 'crop':
            crops = self.database.get_most_common_crops(limit=10)
            return "Which crop are you working with?\n" + "\n".join(f"- {c}" for c in crops)
        if uncertain_about == 'pest_name':
            if not session.crop:
                return "I need to know your crop first. What crop are you growing?"
            solutions = self.database.get_solutions(session.crop)
            return f"Here are all solutions for {session.crop}:\n\n{self.format_solution_response(solutions, session.crop)}"
        return "Could you provide more details about your problem?"

    def _handle_denial(self, session: SessionState) -> str:
        return "Okay — please tell me what you meant or choose from the available options."

    def _handle_confirmation(self, session: SessionState) -> str:
        if session.is_complete():
            solutions = self.database.get_solutions(session.crop, session.pest_name, session.application_type)
            return self.format_solution_response(solutions, session.crop, session.pest_name)
        return self._ask_next_question(session)

    def _handle_provide_info(self, intent: Dict, session: SessionState) -> str:
        """Handle provide_info: update session with any entities and respond or ask next question."""
        entities = intent.get('entities') or {}

        # Crop
        crop_ent = entities.get('crop')
        if crop_ent:
            matched = self.database.fuzzy_match_crop(crop_ent)
            if matched:
                # if different crop selected, reset dependent context
                if session.crop and matched != session.crop:
                    session.crop = matched
                    session.pest_name = None
                    session.application_type = None
                else:
                    session.crop = matched

        # Pest
        pest_ent = entities.get('pest_name')
        if pest_ent:
            matched_pest = self.database.fuzzy_match_pest(pest_ent, session.crop) if session.crop else self.database.fuzzy_match_pest(pest_ent, None)
            if matched_pest:
                session.pest_name = matched_pest

        # Application type
        app_ent = entities.get('application_type')
        if app_ent:
            matched_app = self.database.fuzzy_match_application_type(app_ent)
            if matched_app:
                session.application_type = matched_app

        # If we have enough info, provide solutions
        if session.is_complete():
            solutions = self.database.get_solutions(session.crop, session.pest_name, session.application_type)
            if not solutions:
                return f"No solutions found for {session.crop} - {session.pest_name}."
            return self.format_solution_response(solutions, session.crop, session.pest_name)

        # Otherwise, ask next question
        return self._ask_next_question(session)

    # def _handle_general_question(self, query: str, session: SessionState) -> str:
    #     """Use TF-IDF RAG fallback to answer general questions."""
    #     docs = self._semantic_search(query, top_k=5)
    #     if not docs:
    #         return "I couldn't find relevant information in the knowledge base. Please specify crop or pest."
    #     # Return the retrieved context as answer (or summarize if you add an LLM)
    #     return "Relevant entries from knowledge base:\n\n" + "\n\n".join(docs)

    def _handle_general_question(self, query: str, session: SessionState) -> str:
        """Use TF-IDF RAG + Groq LLM for general question answering."""
    
        # Step 1: Retrieve top matching documents
        docs = self._semantic_search(query, top_k=5)
        if not docs:
         return "I couldn't find relevant information in the knowledge base. Please specify crop or pest."

        # Combine retrieved docs into a context string
        context = "\n\n".join(docs)

        # Step 2: Generate answer using Groq LLM
        prompt = f"""
        You are an agricultural expert. Answer the user's question using ONLY the information provided below.
        If information is missing, say it clearly.

      User question:
      {query}

      Knowledge Base Context:
      {context}

     Provide a clear, helpful answer:
    
     """
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }

        resp = requests.post(self.groq_url, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Groq general question error: {resp.status_code} → {resp.text}")
            return "I encountered an error while trying to answer your question."
        

        llm_answer = resp.choices[0].message["content"]

        return llm_answer

    

    # ---------------------- Conversation Flow Helpers ----------------------
    def _ask_next_question(self, session: SessionState) -> str:
        if not session.crop:
            crops = self.database.get_most_common_crops(limit=8)
            session.pending_question = 'awaiting_crop'
            return "Which crop are you working with?\n" + "\n".join(f"- {c}" for c in crops)

        if not session.pest_name:
            pests = self.database.get_pests(session.crop)
            if not pests:
                return f"No pest data available for {session.crop}."
            if len(pests) == 1:
                session.pest_name = pests[0]
                return self._provide_final_solution(session)
            session.pending_question = 'awaiting_pest'
            return f"What pest or disease in {session.crop}?\n" + "\n".join(f"- {p}" for p in pests)

        return self._provide_final_solution(session)

    def _provide_final_solution(self, session: SessionState) -> str:
        solutions = self.database.get_solutions(session.crop, session.pest_name, session.application_type)
        if not solutions:
            return f"No solutions found for {session.crop} - {session.pest_name}."
        return self.format_solution_response(solutions, session.crop, session.pest_name)

    # ---------------------- Main Query Handler ----------------------
    def handle_query(self, query: str, session: SessionState) -> str:
        """
        Main query handler - orchestrates the entire flow
        """
        # Step 1: Correct typos (uses your Groq-based TextCorrector)
        corrected_query = self.text_corrector.correct_text(query)
        if corrected_query != query:
            logger.info(f"Corrected: '{query}' → '{corrected_query}'")

        # Step 2: Detect intent (Groq with fallback)
        intent = self.detect_intent(corrected_query, session)

        # Step 3: Validate intent
        intent = self._validate_intent(intent)

        logger.info(f"Intent: {intent['intent_type']}, Entities: {intent.get('entities', {})}")

        # If entities present, update session immediately
        entities = intent.get('entities') or {}
        if entities.get('crop'):
            session.crop = self.database.fuzzy_match_crop(entities['crop']) or session.crop
        if entities.get('pest_name'):
            session.pest_name = self.database.fuzzy_match_pest(entities['pest_name'], session.crop) or session.pest_name
        if entities.get('application_type'):
            session.application_type = self.database.fuzzy_match_application_type(entities['application_type']) or session.application_type

        # Route based on intent
        itype = intent['intent_type']
        if itype == 'greeting':
            return self._handle_greeting()
        if itype == 'farewell':
            return self._handle_farewell()
        if itype == 'unclear':
            return self._handle_unclear()
        if itype == 'off_topic':
            return self._handle_off_topic()
        if itype == 'uncertainty':
            return self._handle_uncertainty(intent, session)
        if itype == 'denial':
            return self._handle_denial(session)
        if itype == 'confirmation':
            return self._handle_confirmation(session)
        if itype == 'provide_info':
            return self._handle_provide_info(intent, session)
        if itype == 'question':
            return self._handle_general_question(corrected_query, session)
        if itype == 'list_pest_problem':
            # If intent explicitly asks for list of pests
            crop = entities.get('crop') or session.crop
            if not crop:
                return "Which crop would you like the pest list for?"
            pests = self.database.get_pests(crop)
            if not pests:
                return f"No pest data available for {crop}."
            return f"Pest problems for {crop}:\n" + "\n".join(f"- {p}" for p in pests)

        # default
        return "I'm not sure how to help with that. Could you please clear the crop and problem ?"

    # ---------------------- Chat Entry Point ----------------------
    def chat(self, query: str, session: SessionState) -> Tuple[str, SessionState]:
        session.add_message('user', query)
        response = self.handle_query(query, session)
        session.add_message('assistant', response)
        return response, session
