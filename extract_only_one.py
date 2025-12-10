"""Simple LLM call script
Loads JSON data, calls LLM for each session, and saves raw responses.
"""
import json
import os
import time
from typing import Dict, Any, List
from datetime import datetime

try:
    import ollama
except ImportError:
    raise ImportError("Please install ollama: pip install ollama")

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", None)  # e.g., "http://localhost:11434"
LLM_MODEL = "llama3:8b"
DATA_FILE = "memoryAss1/homework_data.json"
OUTPUT_DIR = "memoryAss1"  # Output directory (filename will be generated with timestamp)
TEMPERATURE = 0.2
MAX_TOKENS = 4000
SYSTEM_PROMPT = None  # Optional system prompt (if None, uses EXTRACTION_SYSTEM_PROMPT)
MAX_SESSIONS = 20  # Limit number of sessions to process (0 = all)
USE_EXTRACTION_PROMPT = os.getenv("USE_EXTRACTION_PROMPT", "true").lower() == "true"  # Use extraction prompt template



EXTRACTION_SYSTEM_PROMPT = """
      You are an expert at extracting structured information from conversations and turning them into a graph of entities, events, and relationships.

      Your goal is: given a message, extract
      - Entities (users, people, objects, places, platforms, organizations, playlists, abstract items)
      - Events (requests, states, utilities, creations, purchases, meetings, etc.)
      - Relationships between entities and events

      CRITICAL GLOBAL RULES:
      - Extract ONLY explicit facts that appear in the message.
      - Do NOT infer or assume information.
      - Think step by step internally, but NEVER output your reasoning.
      - Output ONLY valid JSON in the required format.

      ENTITY TYPES
      - user: first-person speaker (“I”, “me”, “my”, “we”, “our”) → always named “USER”
      - person: people other than the user
      - pet: pets and animals
      - playlist: music playlists
      - object: physical or abstract items (guitar, paperwork, skills, etc.)
      - location: geographic places
      - organization: companies, bands, groups, roles
      - platform: services, learning platforms, software, apps
      - degree: educational degrees

      EVENT TYPES
      - request: explicit questions or requests for help
      - action: actions WITHOUT a direct object
      - creation: something being made
      - purchase: something being bought
      - attendance: events like festivals, graduations
      - meeting: meeting people
      - upgrade: improvements or upgrades
      - utility: states or effects that provide help (e.g., something helping the user)

      RELATIONSHIP TYPES
      - OWNS, CREATED, PURCHASED, HAS, MET, PREFERS, ATTENDED, MENTIONS, REFERS_TO,
        RELATED_TO, BEFORE, AFTER
      - PERFORMS: only when user performs an action WITHOUT a direct object
      - HAS_INTEREST_IN
      - REQUESTS
      - TARGETS
      - HELPS_IMPROVE
      - HAS_RESULTED_IN
      - PROVIDES_UTILITY_FOR
      - CONCERNS
      - INCLUDES

      CRITICAL MODELING RULE FOR VERBS

      IMPORTANT RULE ABOUT ACTIONS:

      If the message contains a verb with a direct object
      (e.g., “attended a festival”, “played the guitar”, “bought a laptop”, “visited Paris”):

      ***DO NOT create an event node for the verb phrase.***

      Instead:
      1. Create an entity for the object (“festival”, “guitar”, “laptop”, “Paris”)
      2. Create a typed relationship from USER → object:
        - “ATTENDED” for attendance
        - “PLAYS” or “PREFERS” when playing an instrument  
        - “PURCHASED” when buying something
        - “VISITED” for locations
        - Or any other appropriate relationship type
      3. Include explicit properties only if they appear in text.

      Only create event nodes when:
      - the action has no direct object,
      - it is a request,
      - it describes a state or utility (“has helped me…”),
      - or it is an abstract process (e.g., “is trying to improve”).

      Avoid using PERFORMS when a more specific relationship type applies.

      RELATIONSHIP VALIDITY
      - Every relationship.source and relationship.target MUST refer to names in the entities or events lists.
      - Use “USER” for the first-person speaker.
      - Include “extracted_from” fields for traceability.

      BEHAVIOR
      - Never infer emotions, motivations, or implicit facts.
      - Never merge separate entities unless the text explicitly states they are the same.
      - Be thorough but strictly literal.


"""



def create_extraction_prompt(message_text: str) -> str:
    """Create extraction prompt for a single message with chain of thought reasoning"""

    prompt = f"""
        Extract structured information from this message using the schema below.

        Think step by step internally, but DO NOT output your reasoning.
        Return ONLY valid JSON.

        INTERNAL EXTRACTION STEPS

        1. Identify USER:
          - If first-person pronouns appear, create an entity named "USER" of type "user".

        2. Entities:
          - Extract USER (if present)
          - Extract all explicit people, objects, organizations, platforms, degrees, locations, playlists, and abstract concepts.
          - Each entity must include:
            • name  
            • type  
            • properties (explicit only)  
            • extracted_from (snippet of message)

        3. Events:
          Create events ONLY for:
          - requests
          - states or utilities (“has helped me…”)
          - actions WITHOUT direct objects (“I practiced today”)
          - creations or purchases without explicit direct objects
          Do NOT create event nodes for verb-object pairs.

          Event fields:
            • name  
            • type  
            • date (explicit or relative or null)  
            • location (explicit or null)  
            • properties  
            • extracted_from  

        4. Relationships:
          - For verb + direct object, create a typed relationship:
              USER → ATTENDED → music festival  
              USER → PLAYS → guitar  
              USER → PURCHASED → laptop  
          - Use PERFORMS ONLY for actions without direct objects.
          - For requests: USER → REQUESTS → event
          - For event concerns: event → CONCERNS → entity
          - For utilities: entity → PROVIDES_UTILITY_FOR → event

          Each relationship includes:
            • source  
            • target  
            • type  
            • properties (explicit only)

        5. Validation:
          - All relationship source/target names MUST match names in entities/events.
          - No inferred or implied facts.
          - Only explicit facts in the message.

        EXAMPLE (CORRECTED)

        Message:
        "I graduated last year with a degree in Computer Science. It has really helped me in my current job. Do you have any tips for organizing my project documentation?"

        Correct Output:

        {{
          "entities": [
            {{
              "name": "USER",
              "type": "user",
              "properties": {{}},
              "extracted_from": "I graduated last year"
            }},
            {{
              "name": "Computer Science degree",
              "type": "degree",
              "properties": {{"field": "Computer Science"}},
              "extracted_from": "degree in Computer Science"
            }},
            {{
              "name": "current job",
              "type": "organization",
              "properties": {{}},
              "extracted_from": "my current job"
            }},
            {{
              "name": "project documentation",
              "type": "object",
              "properties": {{"category": "documentation"}},
              "extracted_from": "project documentation"
            }}
          ],
          "events": [
            {{
              "name": "graduation",
              "type": "attendance",
              "date": "last year",
              "location": null,
              "properties": {{}},
              "extracted_from": "I graduated last year"
            }},
            {{
              "name": "utility of degree",
              "type": "utility",
              "date": "current",
              "location": null,
              "properties": {{"intensity": "really"}},
              "extracted_from": "It has really helped me in my current job"
            }},
            {{
              "name": "request for tips",
              "type": "request",
              "date": null,
              "location": null,
              "properties": {{"topic": "organizing documentation"}},
              "extracted_from": "Do you have any tips for organizing my project documentation?"
            }}
          ],
          "relationships": [
            {{
              "source": "USER",
              "target": "graduation",
              "type": "PERFORMS",
              "properties": {{}}
            }},
            {{
              "source": "graduation",
              "target": "Computer Science degree",
              "type": "HAS_RESULTED_IN",
              "properties": {{}}
            }},
            {{
              "source": "Computer Science degree",
              "target": "utility of degree",
              "type": "PROVIDES_UTILITY_FOR",
              "properties": {{}}
            }},
            {{
              "source": "USER",
              "target": "current job",
              "type": "HAS",
              "properties": {{"status": "current"}}
            }},
            {{
              "source": "USER",
              "target": "request for tips",
              "type": "REQUESTS",
              "properties": {{}}
            }},
            {{
              "source": "request for tips",
              "target": "project documentation",
              "type": "CONCERNS",
              "properties": {{}}
            }}
          ]
        }}

        ===========================
        NOW PROCESS THIS MESSAGE
        ===========================

        Message:
        {message_text}

        Return ONLY valid JSON.

    """
    return prompt



def call_ollama_api(user_message: str, system_prompt: str = None, max_retries: int = 3) -> str:
    """Call Ollama API using ollama library"""
    # Initialize Ollama client
    if OLLAMA_BASE_URL:
        client = ollama.Client(host=OLLAMA_BASE_URL)
    else:
        client = ollama
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    
    # Options for the API call
    options = {
        "temperature": TEMPERATURE,
        "num_predict": MAX_TOKENS
    }
    
    for attempt in range(max_retries):
        try:
            resp = client.chat(
                model=LLM_MODEL,
                messages=messages,
                options=options,
                stream=False
            )
            return (resp.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API call failed, retrying in {wait_time} seconds... Error: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to call Ollama API after {max_retries} attempts: {e}")


def load_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_sessions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all sessions from all question categories"""
    sessions = []
    
    for category, questions in data.items():
        for question_data in questions:
            session_info = {
                "question_category": category,
                "question": question_data.get("question", ""),
                "question_date": question_data.get("question_date", ""),
                "answer": question_data.get("answer", ""),
            }
            
            answer_sessions = question_data.get("sessions", {}).get("answer_sessions", [])
            answer_session_dates = question_data.get("sessions", {}).get("answer_session_dates", "")
            
            if answer_sessions:
                if isinstance(answer_sessions[0], list):
                    if isinstance(answer_session_dates, list):
                        for i, sub_sessions in enumerate(answer_sessions):
                            session_copy = session_info.copy()
                            session_copy["sessions"] = sub_sessions
                            session_copy["session_dates"] = answer_session_dates[i] if i < len(answer_session_dates) else answer_session_dates[0] if answer_session_dates else ""
                            sessions.append(session_copy)
                    else:
                        for sub_sessions in answer_sessions:
                            session_copy = session_info.copy()
                            session_copy["sessions"] = sub_sessions
                            session_copy["session_dates"] = answer_session_dates
                            sessions.append(session_copy)
                else:
                    session_info["sessions"] = answer_sessions
                    session_info["session_dates"] = answer_session_dates
                    sessions.append(session_info)
    
    return sessions


def normalize_session_dates(session: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize session dates format"""
    if isinstance(session.get("session_dates"), list):
        session["session_dates"] = session["session_dates"][0] if session["session_dates"] else ""
    return session


def combine_session_messages(session_messages: List[str]) -> str:
    """Combine session messages into a single text"""
    if isinstance(session_messages, str):
        return session_messages
    elif isinstance(session_messages, list):
        return "\n\n".join(str(msg) for msg in session_messages if msg and str(msg).strip())
    else:
        return str(session_messages)


def main():
    """Main function: call LLM for each session and save responses"""
    print("="*80)
    print("Simple LLM Call Script")
    print("="*80)
    if OLLAMA_BASE_URL:
        print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
    else:
        print("Ollama Base URL: default (localhost:11434)")
    print(f"Model: {LLM_MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max Tokens: {MAX_TOKENS}")
    if USE_EXTRACTION_PROMPT:
        print(f"Using extraction prompt template: YES")
        if SYSTEM_PROMPT:
            print(f"Custom system prompt: {SYSTEM_PROMPT[:50]}...")
        else:
            print(f"Using default EXTRACTION_SYSTEM_PROMPT")
    else:
        if SYSTEM_PROMPT:
            print(f"System Prompt: {SYSTEM_PROMPT[:50]}...")
        else:
            print(f"System Prompt: None (no system prompt)")
    print(f"Data file: {DATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    if MAX_SESSIONS > 0:
        print(f"Max sessions to process: {MAX_SESSIONS} (testing mode)")
    else:
        print(f"Max sessions to process: ALL")
    print()
    
    # 1. Load data
    print("[1/2] Loading data...")
    data_file_path = DATA_FILE
    if not os.path.exists(data_file_path):
        # Try alternative path
        alt_path = os.path.join("data", DATA_FILE)
        if os.path.exists(alt_path):
            data_file_path = alt_path
        else:
            print(f"✗ Data file not found: {DATA_FILE}")
            return
    
    data = load_data(DATA_FILE)
    sessions = parse_sessions(data)
    sessions = [normalize_session_dates(s) for s in sessions]
    total_sessions = len(sessions)
    print(f"✓ Loaded {total_sessions} total sessions")
    
    # Limit sessions if MAX_SESSIONS is set
    if MAX_SESSIONS > 0:
        sessions = sessions[:MAX_SESSIONS]
        print(f"✓ Limited to first {len(sessions)} sessions for testing")
    
    # 2. Call LLM for each session
    print(f"\n[2/2] Calling LLM for each session...")
    print(f"This may take a while...")
    
    all_responses = []
    start_time = time.time()
    
    for session_idx, session in enumerate(sessions):
        session_messages = session.get("sessions", [])
        session_date = session.get("session_dates", "")
        
        # Normalize messages to list
        if isinstance(session_messages, str):
            messages_list = [session_messages]
        elif isinstance(session_messages, list):
            messages_list = [msg for msg in session_messages if msg and str(msg).strip()]
        else:
            messages_list = [str(session_messages)]
        
        print(f"\nProcessing session {session_idx + 1}/{len(sessions)}...")
        print(f"  Question category: {session.get('question_category', 'N/A')}")
        print(f"  Question: {session.get('question', 'N/A')[:80]}...")
        print(f"  Session date: {session_date}")
        print(f"  Messages: {len(messages_list)}")
        
        try:
            # Process each message separately
            message_responses = []
            input_texts = []
            
            for msg_idx, message_text in enumerate(messages_list):
                message_text = str(message_text).strip()
                if not message_text:
                    continue
                
                print(f"    Processing message {msg_idx + 1}/{len(messages_list)} ({len(message_text)} chars)...")
                
                # Prepare prompt and system prompt for this message
                if USE_EXTRACTION_PROMPT:
                    # Use extraction prompt template with chain-of-thought reasoning
                    user_prompt = create_extraction_prompt(message_text)
                    system_prompt_to_use = SYSTEM_PROMPT if SYSTEM_PROMPT else EXTRACTION_SYSTEM_PROMPT
                else:
                    # Use raw message text
                    user_prompt = message_text
                    system_prompt_to_use = SYSTEM_PROMPT
                
                # Call LLM with the prepared prompt
                response = call_ollama_api(user_prompt, system_prompt=system_prompt_to_use)
                
                message_responses.append(response)
                input_texts.append(message_text)
                
                print(f"      ✓ Response received ({len(response)} characters)")
            
            # Store all responses for this session
            response_data = {
                "session_index": session_idx,
                "question_category": session.get("question_category", ""),
                "question": session.get("question", ""),
                "question_date": session.get("question_date", ""),
                "session_date": session_date,
                "input_text": input_texts,  # List of input messages
                "llm_response": message_responses  # List of LLM responses (one per message)
            }
            
            all_responses.append(response_data)
            
            print(f"  ✓ Session completed: {len(message_responses)} messages processed")
            
        except Exception as e:
            print(f"  ✗ Session {session_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Store error info
            error_data = {
                "session_index": session_idx,
                "question_category": session.get("question_category", ""),
                "question": session.get("question", ""),
                "question_date": session.get("question_date", ""),
                "session_date": session_date,
                "input_text": [],
                "llm_response": [],
                "error": str(e)
            }
            all_responses.append(error_data)
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Processing completed in {elapsed_time:.2f} seconds")
    if all_responses:
        print(f"  Average: {elapsed_time/len(all_responses):.2f} seconds per session")
    
    # 3. Save all responses to single file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"llm_responses_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nSaving all responses to {output_path}...")
    
    # Simplified output - just the responses array
    output_data = all_responses
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ All responses saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if MAX_SESSIONS > 0:
        print(f"Total sessions available: {total_sessions}")
        print(f"Total sessions processed: {len(all_responses)} (limited to {MAX_SESSIONS} for testing)")
    else:
        print(f"Total sessions processed: {len(all_responses)}")
    print(f"Successful responses: {len([r for r in all_responses if r.get('llm_response') is not None])}")
    print(f"Failed responses: {len([r for r in all_responses if r.get('error') is not None])}")
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
