# Memory Extraction and Neo4j Graph Pipeline

A complete pipeline for extracting structured memories from conversational data using LLMs and storing them in Neo4j graph database.

## Overview

This project consists of two main phases:

1. **Extraction Phase**: Uses LLM (Llama3 8B via Ollama) to extract entities, events, and relationships from conversation sessions
2. **Neo4j Loading Phase**: Loads extracted data into Neo4j, displays graph structure, and visualizes the knowledge graph

## Project Architecture

```
homework_data.json (conversation sessions)
    ↓
[1] Extraction Phase (extract_only_one_message_per_call.py)
    - Calls LLM for each message/session
    - Extracts: entities, events, relationships
    - Output: llm_responses_split.json
    ↓
[2] Neo4j Loading Phase (neo4j notebook.ipynb)
    - Parses LLM response files
    - Creates graph schema in Neo4j
    - Loads sessions (each session = unique user)
    - Visualizes graph structure
```

---

## Phase 1: LLM Extraction

### Files

- **`extract_only_one_message_per_call.py`**: Standalone Python script for extraction (should be used in cluster)

### Purpose

Extracts structured information from conversational data:
- **Entities**: People, objects, places, organizations, etc.
- **Events**: Actions, occurrences, requests, etc.
- **Relationships**: Connections between User, Entities, and Events

**Key Features**:
- Processes each message separately for better context
- Saves responses in structured JSON format
- Handles errors and retries automatically


Each session entry contains:
```json
{
  "session_index": 0,
  "session_date": "2023/05/21 (Sun) 13:13",
  "llm_response": [
    {
      "entities": [...],
      "events": [...],
      "relationships": [...]
    },
    ...
  ]
}
```

---

## Phase 2: Neo4j Loading and Visualization

### File

- **`neo4j notebook.ipynb`**: Google Colab notebook for Neo4j operations

### Purpose

Loads extracted data into Neo4j graph database and provides visualization:
- Parses LLM response files
- Creates Neo4j schema (constraints, indexes)
- Loads each session as a unique user
- Displays graph structure (text format)
- Visualizes graph using matplotlib and NetworkX

### Key Features

- **Session-based User Creation**: Each session gets its own unique user node (`user_session_{index}`)
- **Automated Pipeline**: Single function to load, display, and visualize
- **Graph Visualization**: Interactive graph visualization with NetworkX and matplotlib
- **Statistics**: Summary statistics of nodes and relationships

### Usage

#### 1. Setup Configuration

```python
NEO4J_URI = "bolt://your-neo4j-server:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"
```

#### 2. Upload LLM Response File

Use the file upload cell to upload your extraction output file:
- `llm_responses_split.json`

#### 3. Parse Uploaded File

The notebook automatically:
- Detects file format
- Parses JSON responses
- Handles malformed JSON gracefully
- Combines multiple responses per session if needed

#### 4. Load Session to Neo4j

**Individual Session Loading**:
```python
result = load_session_to_neo4j(session_id=0)
# Loads session at index 0 into Neo4j
```

**Automated Pipeline** (Load + Display + Visualize):
```python
pipeline_load_and_visualize(session_id=0)
# Complete pipeline: loads session, displays structure, visualizes graph
```

### Main Functions

#### `load_session_to_neo4j(session_id, ...)`

Loads a specific session into Neo4j.

**Parameters**:
- `session_id`: Session index (int) or session_index value
- `parsed_sessions`: List of parsed sessions (optional, uses global if available)
- `neo4j_uri`, `neo4j_username`, `neo4j_password`: Neo4j connection details (optional, uses global config)

**Returns**: Dictionary with success status, user_id, and counts

**Example**:
```python
result = load_session_to_neo4j(0)
if result['success']:
    print(f"Loaded {result['entities']} entities, {result['events']} events")
```

#### `pipeline_load_and_visualize(session_id, ...)`

Complete automated pipeline:
1. Loads session into Neo4j
2. Displays graph structure (users, entities, events, relationships)
3. Visualizes graph with matplotlib/NetworkX

**Parameters**:
- `session_id`: Session index to process
- `show_structure`: Whether to print graph structure (default: True)
- `show_visualization`: Whether to show graph visualization (default: True)
- Other parameters same as `load_session_to_neo4j`

**Example**:
```python
# Load session 0, display structure, and visualize
pipeline_load_and_visualize(0)

# Load session 5 without visualization
pipeline_load_and_visualize(5, show_visualization=False)
```

### Graph Schema

**Node Types**:
- **User**: Each session gets a unique user node (`user_session_{index}`)
- **Entity**: People, objects, places, organizations, etc.
- **Event**: Actions, occurrences, requests, etc.

**Relationships**:
- User → Entity/Event: `PERFORMS`, `REQUESTS`, `HAS`, `MENTIONS`, etc.
- Entity → Entity: Various relationship types
- Event → Event: Temporal and causal relationships
- Event → Entity: Involvement relationships

### Visualization

The visualization displays:
- **Nodes**: Color-coded by type (User=red, Entity=cyan, Event=green)
- **Edges**: Directed relationships between nodes
- **Labels**: Node names (truncated if long)
- **Edge Labels**: Relationship types (limited to first 30 for clarity)

### Additional Features

#### Graph Statistics

Run the statistics cell to see:
- Total counts by node type
- Relationship counts by type
- Entity/Event type distributions

#### Delete Operations

Cells are provided to:
- Delete all data from Neo4j (reset database)
- Delete specific session data (by user_id)

---

## Complete Workflow

### Step 1: Extract with LLM

1. Run the python file in cluster with sbatch file
2. Get the output file: `llm_responses_split.json`

### Step 2: Load to Neo4j

1. Open `neo4j notebook.ipynb` in Google Colab
2. Configure Neo4j connection settings
3. Upload the `llm_responses_split.json` file
4. Run parsing cells
5. Load sessions using `pipeline_load_and_visualize(session_id)`

### Example Workflow

```python
# In neo4j notebook.ipynb

# 1. Load session 0
pipeline_load_and_visualize(0)

# 2. Load session 5
pipeline_load_and_visualize(5)

# 3. View statistics
# (Run the statistics cell)

# 4. Load another session
pipeline_load_and_visualize(10)
```

---

## Dependencies

### Neo4j Phase (`neo4j notebook.ipynb`)
- `neo4j==5.15.0`: Neo4j Python driver
- `networkx`: Graph visualization
- `matplotlib`: Plotting
- `requests==2.31.0`: HTTP requests
- Google Colab environment

---

## Configuration

### Neo4j Configuration

In `neo4j notebook.ipynb`:
- `NEO4J_URI`: Neo4j connection URI (bolt://host:port)
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

---

## Output Files

### Extraction Outputs

- `llm_responses_split.json`: One LLM response per message per session


### Format Structure

```json
[
  {
    "session_index": 0,
    "session_date": "2023/05/21 (Sun) 13:13",
    "llm_response": [
      {
        "entities": [
          {
            "name": "guitar",
            "type": "object",
            "properties": {},
            "extracted_from": "..."
          }
        ],
        "events": [...],
        "relationships": [...]
      }
    ]
  }
]
```

---

## Important Notes

### Session Management

- Each session is treated as a **unique user** in Neo4j
- User ID format: `user_session_{session_index}`
- This allows tracking which session created which relationships
- Entities and Events may be shared across sessions (merged by name+type)

### Data Persistence

- Entities and Events are merged if they have the same name and type
- Relationships are session-specific (linked to the session's user)
- You can load multiple sessions sequentially into the same Neo4j database

### Error Handling

- JSON parsing errors are handled gracefully (empty arrays returned)
- Network errors include retry logic
- Neo4j connection errors are caught and reported

---

## Troubleshooting

### Extraction Issues

- **Model not found**: Ensure Ollama model is installed (`ollama pull llama3.1:8b`)
- **JSON parsing errors**: Check LLM responses - some may be malformed

### Neo4j Issues

- **Connection failed**: Verify URI, username, and password
- **No nodes/edges in visualization**: Ensure session was loaded successfully
- **Empty graph**: Check that relationships exist in the extracted data

---

## Project Files

```
project/
├── neo4j notebook.ipynb               # Neo4j loading and visualization (Colab)
├── extract_only_one_message_per_call.py  # Standalone extraction script
├── homework_data.json                 # Input conversation data
├── llm_responses_split.json           # Extraction output (split format)
└── README.md                          # This file
```
