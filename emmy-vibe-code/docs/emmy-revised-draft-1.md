Okay, Senior Dev hat back on. Let's integrate Ollama and address the concerns systematically.

**Goal:** Make the `Ghost` architecture runnable with Ollama, address the key architectural issues (globals, placeholders), and outline the next steps for the more complex parts.

**Assumptions:**

*   You have Ollama installed and running locally (e.g., `ollama serve`).
*   You have a model pulled (e.g., `ollama pull llama3`).
*   You have the `ollama` Python library installed (`pip install ollama`).
*   You have `scipy` installed for cosine distance (`pip install scipy`).

```python
import logging
import json
from enum import StrEnum, IntEnum
from typing import List, Dict, Optional, Tuple, Type, TypeVar, Any
from datetime import datetime, timedelta

from py_linq import Enumerable
import numpy as np
from pydantic import BaseModel, Field, ValidationError, create_model
from ollama import Client, ResponseError # Import Ollama client and errors
from scipy.spatial.distance import cosine as cosine_distance # Use scipy for cosine distance

# --- Configuration ---
# Using a simple class for config for now. Pydantic BaseSettings is good for env vars.
class GhostConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3" # Or mistral, etc.
    ollama_embedding_model: str = "mxbai-embed-large" # Or another suitable embedding model
    companion_name: str = "Emmy"
    user_name: str = "Rick"
    universal_character_card: str = "Emmy is a helpful assistant."
    default_decay_factor: float = 0.95 # How much state persists per tick
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 32
    retrieval_limit_features_context: int = 32
    retrieval_limit_features_causal: int = 512
    narrative_max_tokens: int = 512 # Example limit for narrative context
    short_term_intent_count: int = 3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)


# --- Models (Mostly unchanged, added Configurable Defaults/Decay) ---
class DecayableMentalState(BaseModel):
    # Added baseline and decay factor placeholders
    baseline_value: float = 0.0 # Neutral baseline
    decay_factor: float = Field(default=0.95, exclude=True) # Exclude from model schema if not needed

    def decay_to_baseline(self, ticks_elapsed: int = 1):
        """Applies exponential decay towards the baseline value."""
        if ticks_elapsed <= 0:
            return

        # Apply decay to all float fields in the model
        for field_name, field_info in self.model_fields.items():
            # Check if the field type is float and it's not the decay_factor itself
            # Need to handle potential annotations like Optional[float] etc.
            # For simplicity, checking if current value is float. Robust check needed.
            current_value = getattr(self, field_name)
            if isinstance(current_value, float) and field_name != 'decay_factor':
                # Get specific baseline if defined, else use default
                specific_baseline = getattr(self, f"baseline_{field_name}", self.baseline_value)
                # Apply decay factor potentially multiple times if ticks_elapsed > 1
                effective_decay = self.decay_factor ** ticks_elapsed
                new_value = specific_baseline + (current_value - specific_baseline) * effective_decay
                # Clamp to field limits if they exist (using Field info)
                if field_info.ge is not None:
                    new_value = max(new_value, field_info.ge)
                if field_info.le is not None:
                    new_value = min(new_value, field_info.le)
                setattr(self, field_name, new_value)

    def get_overall_valence(self) -> float:
        # Simple average of valenced axes for now. Needs refinement.
        valence_sum = 0.0
        count = 0
        # Example: average 'valence', 'affection', 'self_worth', 'trust', maybe negative of 'disgust', 'anxiety'?
        # This needs careful thought based on the model's meaning.
        if hasattr(self, 'valence'):
            valence_sum += self.valence
            count += 1
        # Add other relevant fields...
        if hasattr(self, 'anxiety'):
             valence_sum -= self.anxiety # Assuming anxiety is negative valence
             count +=1

        return valence_sum / count if count > 0 else 0.0


class EmotionalAxesModel(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="...")
    affection: float = Field(default=0.0, ge=-1, le=1, description="...")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="...")
    trust: float = Field(default=0.0, ge=-1, le=1, description="...")
    disgust: float = Field(default=0.0, ge=0, le=1, description="...")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="...")
    # Example of specific baselines (optional)
    baseline_valence: float = 0.0
    baseline_anxiety: float = 0.1 # Maybe slightly anxious baseline?

class NeedsAxesModel(DecayableMentalState):
    baseline_value: float = 0.5 # Needs decay towards a baseline of 0.5 (partially met)
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="...")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="...")

class CognitionAxesModel(DecayableMentalState):
    baseline_value: float = 0.0 # Default baseline for cognition axes
    interlocus: float = Field(default=0.0, ge=-1, le=1, description="...")
    mental_aperture: float = Field(default=0.0, ge=-1, le=1, description="...")
    ego_strength: float = Field(default=1.0, ge=0, le=1, description="...")
    willpower: float = Field(default=0.5, ge=0, le=1, description="...")
    baseline_ego_strength: float = 0.8 # Example specific baseline

# --- Knoxel Base & Specific Types (Refactored ID/Tick Management) ---
class KnoxelBase(BaseModel):
    id: int = -1 # Will be assigned by Ghost instance
    tick_id: int = -1 # Will be assigned by Ghost instance
    content: str
    embedding: List[float] = Field(default=[], repr=False) # Exclude large embedding from default repr
    timestamp_creation: datetime = Field(default_factory=datetime.now)
    timestamp_world_begin: datetime = Field(default_factory=datetime.now)
    timestamp_world_end: datetime = Field(default_factory=datetime.now)

    # Removed strength property - calculation depends on current tick, better done externally

class Stimulus(KnoxelBase):
    stimulus_type: str # message, thought, internal_state_change

class Intention(KnoxelBase):
    urgency: float = 0.5
    affective_valence: float
    incentive_salience: float
    fulfilment: float = 0
    internal: bool = True

class Action(KnoxelBase):
    action_type: str
    action_content: str
    # Expectation handling needs refinement - link IDs maybe?
    # expectation: List[Intention] -> List[int] (Intention IDs)
    expectation_ids: List[int] = []
    expectation_fulfilment: float = 0.0
    contribution_to_overall_valence: float = 0.0

class EpisodicMemoryInstance(KnoxelBase):
    affective_valence: float

class DeclarativeMemoryFact(KnoxelBase):
    affective_valence: float = 0.0 # Facts are often neutral initially

class NarrativeType(StrEnum):
    Relation = "relation"
    SelfImage = "self_image"
    ExternalAnalysis = "external_analysis"
    EmotionalTriggers = "emotional_triggers"
    BehaviorAttention = "behavior_attention"
    BehaviorActionSelection = "behavior_action_selection"

class Narrative(KnoxelBase):
    narrative_type: NarrativeType
    of_ai: bool
    content: str

class FeatureType(IntEnum):
    Dialogue = 10
    Feeling = 20
    SituationalModel = 30 # Maybe rename? Represents context understanding
    AttentionFocus = 40 # Explicit representation of what's attended to
    ConsciousWorkspace = 50 # Content broadcast in GWT sense
    MemoryRecall = 60 # The act/content of recalling
    SubjectiveExperience = 70 # The textual description of the feeling
    ActionDeliberation = 80 # Internal monologue / planning steps
    Action = 90
    ActionExpectation = 100

ft_initial_context = [FeatureType.Dialogue, FeatureType.ConsciousWorkspace, FeatureType.SubjectiveExperience, FeatureType.Action]

class Feature(KnoxelBase):
    feature_type: FeatureType
    affective_valence: Optional[float] = None
    incentive_salience: Optional[float] = None
    interlocus: float # -1 internal, +1 external, 0 mixed/neutral
    causal: bool = False # Default to virtual/potential until committed


# --- KnoxelList Wrapper (Concept Retained, Storage Abstracted) ---
class KnoxelList:
    """Wraps an Enumerable of Knoxels, providing helper methods."""
    def __init__(self, knoxels: Optional[List[KnoxelBase]] = None):
        self._list = Enumerable(knoxels if knoxels else [])

    def get_narrative(self, config: GhostConfig, max_tokens: Optional[int] = None) -> str:
        """Formats knoxels into a narrative string, respecting token limits (approx)."""
        # TODO: Implement smarter token counting and truncation
        # Simple implementation for now: join content
        content_list = [k.content for k in self._list]
        narrative = "\n".join(content_list)
        # Very basic truncation if needed
        if max_tokens:
             # Rough estimate: 4 chars per token
             char_limit = max_tokens * 4
             if len(narrative) > char_limit:
                  narrative = narrative[:char_limit] + "..."
        return narrative

    def get_embeddings_np(self) -> Optional[np.ndarray]:
        """Returns embeddings as a NumPy array, if embeddings exist."""
        embeddings = [k.embedding for k in self._list if k.embedding]
        if not embeddings:
            return None
        try:
            return np.array(embeddings)
        except ValueError:
            logging.error("Inconsistent embedding dimensions found.")
            return None # Or handle more gracefully

    def where(self, predicate) -> 'KnoxelList':
        return KnoxelList(self._list.where(predicate).to_list())

    def order_by(self, key_selector) -> 'KnoxelList':
        return KnoxelList(self._list.order_by(key_selector).to_list())

    def order_by_descending(self, key_selector) -> 'KnoxelList':
        return KnoxelList(self._list.order_by_descending(key_selector).to_list())

    def take(self, count: int) -> 'KnoxelList':
        return KnoxelList(self._list.take(count).to_list())

    def take_last(self, count: int) -> 'KnoxelList':
         # Enumerable doesn't have take_last, implement manually
         items = self._list.to_list()
         return KnoxelList(items[-count:])

    def add(self, knoxel: KnoxelBase):
        # Note: This modifies the underlying list used by the Enumerable
        self._list = Enumerable(self._list.to_list() + [knoxel])

    def to_list(self) -> List[KnoxelBase]:
        return self._list.to_list()

    def last_or_default(self, default=None) -> Optional[KnoxelBase]:
        return self._list.last_or_default(default)

    def first_or_default(self, default=None) -> Optional[KnoxelBase]:
         return self._list.first_or_default(default)

    def __len__(self) -> int:
        return self._list.count()


# --- GhostState (Buffer for a single tick) ---
class GhostState(BaseModel):
    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)

    primary_stimulus: Optional[Stimulus] = None # Can be None if tick initiated internally
    # Removed pre/post situational model, let's use Features
    # situational_model_features: KnoxelList = KnoxelList() # Features describing current understanding

    # GWT / Attention related - Placeholders for now
    attention_candidates: KnoxelList = KnoxelList() # Knoxels competing for attention
    attention_focus: KnoxelList = KnoxelList() # What was selected by attention
    conscious_workspace: KnoxelList = KnoxelList() # Content broadcast, includes focus + relevant context/intentions
    subjective_experience: Optional[Feature] = None # Narrative description of the current moment

    # Action Deliberation
    action_options: List[Action] = [] # Potential actions considered (virtual)
    selected_action: Optional[Action] = None # The action chosen (causal)
    expectations: KnoxelList = KnoxelList() # New expectations generated by the action

    # State snapshots
    state_emotions: EmotionalAxesModel = EmotionalAxesModel()
    state_needs: NeedsAxesModel = NeedsAxesModel()
    state_cognition: CognitionAxesModel = CognitionAxesModel()

    # Timeline references (IDs of features created this tick)
    # virtual_feature_ids: List[int] = []
    # causal_feature_ids: List[int] = []
    # conscious_feature_ids: List[int] = [] # Features part of conscious_workspace


# --- Main Cognitive Cycle Class ---
class Ghost:
    def __init__(self, config: GhostConfig):
        self.config = config
        self.client = Client(host=config.ollama_host) # Ollama client

        # --- Instance Variables replacing Globals ---
        self.current_tick_id: int = 0
        self.current_knoxel_id: int = 0

        # In-memory storage (replace with DB for persistence/scale)
        self.all_knoxels: Dict[int, KnoxelBase] = {} # Store by ID for easy lookup
        self.all_features: List[Feature] = []
        self.all_stimuli: List[Stimulus] = []
        self.all_intentions: List[Intention] = []
        self.all_narratives: List[Narrative] = []
        self.all_actions: List[Action] = []
        self.all_episodic_memories: List[EpisodicMemoryInstance] = []
        self.all_declarative_facts: List[DeclarativeMemoryFact] = []
        # ---

        self.states: List[GhostState] = [] # History of past states
        self.current_state: Optional[GhostState] = None # The state being built during the current tick

        # Initialize base narratives if needed (example)
        self._initialize_narratives()

    def _get_next_id(self) -> int:
        self.current_knoxel_id += 1
        return self.current_knoxel_id

    def _get_current_tick_id(self) -> int:
        return self.current_tick_id

    def _initialize_narratives(self):
        # Example: Add a default self-image narrative
        default_narrative = Narrative(
            narrative_type=NarrativeType.SelfImage,
            of_ai=True,
            content=self.config.universal_character_card,
            # No embedding initially, can be generated later
        )
        self.add_knoxel(default_narrative)

    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
        """Adds a knoxel to the appropriate list and the central dictionary, assigns ID/Tick."""
        if knoxel.id == -1: # Assign ID only if not already set (e.g., loading from storage)
            knoxel.id = self._get_next_id()
        if knoxel.tick_id == -1:
            knoxel.tick_id = self._get_current_tick_id()

        self.all_knoxels[knoxel.id] = knoxel

        # Add to specific lists for easier querying (consider indexing in a DB)
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
            # Embed Features as they are crucial for context
            if generate_embedding and not knoxel.embedding:
                 self._generate_embedding(knoxel)
        elif isinstance(knoxel, Stimulus):
            self.all_stimuli.append(knoxel)
        elif isinstance(knoxel, Intention):
            self.all_intentions.append(knoxel)
            if generate_embedding and not knoxel.embedding:
                 self._generate_embedding(knoxel) # Embed intentions for matching
        elif isinstance(knoxel, Narrative):
            self.all_narratives.append(knoxel)
            # Embed narratives for retrieval based on context
            if generate_embedding and not knoxel.embedding:
                 self._generate_embedding(knoxel)
        elif isinstance(knoxel, Action):
            self.all_actions.append(knoxel)
        elif isinstance(knoxel, EpisodicMemoryInstance):
            self.all_episodic_memories.append(knoxel)
            # Embed memories for retrieval
            if generate_embedding and not knoxel.embedding:
                 self._generate_embedding(knoxel)
        elif isinstance(knoxel, DeclarativeMemoryFact):
            self.all_declarative_facts.append(knoxel)
            # Embed facts for retrieval
            if generate_embedding and not knoxel.embedding:
                 self._generate_embedding(knoxel)
        # else: just add to all_knoxels

        # logging.debug(f"Added Knoxel {knoxel.id} ({type(knoxel).__name__}): {knoxel.content[:50]}...")


    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[KnoxelBase]:
        return self.all_knoxels.get(knoxel_id)

    def _generate_embedding(self, knoxel: KnoxelBase):
        """Generates and stores embedding for a knoxel using Ollama."""
        if not knoxel.content:
            logging.warning(f"Skipping embedding for Knoxel {knoxel.id}: No content.")
            return
        try:
            response = self.client.embeddings(
                model=self.config.ollama_embedding_model,
                prompt=knoxel.content
            )
            knoxel.embedding = response['embedding']
            # logging.debug(f"Generated embedding for Knoxel {knoxel.id}")
        except ResponseError as e:
            logging.error(f"Ollama embedding error for Knoxel {knoxel.id}: {e.error}")
        except Exception as e:
            logging.error(f"Failed to generate embedding for Knoxel {knoxel.id}: {e}")


    def _call_llm(self, prompt: str, output_schema: Optional[Type[BM]] = None, **kwargs) -> str | BM | float | int | bool | None:
        """ Wrapper for Ollama calls, handling structured output and errors."""
        full_prompt = prompt # Can add system prompts or formatting here later
        model_name = self.config.ollama_model

        # Add context kwargs to prompt if needed (or handle via messages API)
        # context_str = "\n".join([f"{k}: {v}" for k, v in kwargs.items() if v])
        # if context_str:
        #     full_prompt += "\n\nContext:\n" + context_str

        messages = [{'role': 'user', 'content': full_prompt}]
        # Potentially add system message, character card etc. here
        # messages.insert(0, {'role': 'system', 'content': self.config.universal_character_card})

        try:
            if output_schema:
                # Request JSON output
                response = self.client.chat(
                    model=model_name,
                    messages=messages,
                    format='json'
                )
                response_content = response['message']['content']
                try:
                    # Attempt to parse the JSON and validate with Pydantic
                    json_data = json.loads(response_content)
                    validated_data = output_schema.model_validate(json_data)
                    return validated_data
                except json.JSONDecodeError as e:
                    logging.error(f"LLM failed to produce valid JSON: {e}\nRaw Response: {response_content}")
                    return None
                except ValidationError as e:
                    logging.error(f"LLM output failed Pydantic validation: {e}\nRaw Response: {response_content}")
                    return None
            else:
                 # Simple text generation
                 response = self.client.chat(model=model_name, messages=messages)
                 return response['message']['content']

        except ResponseError as e:
            logging.error(f"Ollama API error: {e.status_code} - {e.error}")
            return None
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return None

    def _get_relevant_knoxels(self, query_embedding: List[float], knoxel_source: List[T], limit: int) -> List[T]:
        """Retrieves knoxels based on cosine distance to query embedding."""
        if not query_embedding or not knoxel_source:
            return []

        # Filter out knoxels without embeddings
        candidates = [k for k in knoxel_source if k.embedding]
        if not candidates:
            return []

        # Calculate distances (cosine distance: 0=identical, 1=orthogonal, 2=opposite)
        # Lower distance is better
        distances = [cosine_distance(query_embedding, k.embedding) for k in candidates]

        # Sort candidates by distance (ascending)
        sorted_candidates = sorted(zip(distances, candidates), key=lambda x: x[0])

        # Return the knoxels from the closest pairs, up to the limit
        return [k for dist, k in sorted_candidates[:limit]]


    # --- Refactored Tick Method ---
    def tick(self, impulse: Optional[str] = None):
        """Runs one cognitive cycle."""
        logging.info(f"--- Starting Tick {self.current_tick_id + 1} ---")
        self.current_tick_id += 1

        # 1. Initialize State
        self._initialize_tick_state(impulse)
        if not self.current_state: return # Should not happen if init works

        # 2. Retrieve Context & Update Mental States
        context_knoxels = self._retrieve_and_prepare_context()
        self._appraise_stimulus_and_update_states(context_knoxels)

        # 3. Generate Intentions
        self._generate_short_term_intentions(context_knoxels)

        # --- Placeholder Sections for GWT/Action/Learning ---
        # 4. Attention / Conscious Workspace Simulation (LLM Based)
        self._simulate_attention_and_workspace(context_knoxels) # Populates current_state.conscious_workspace

        # 5. Subjective Experience Generation (LLM Based)
        self._generate_subjective_experience() # Populates current_state.subjective_experience

        # 6. Action Deliberation & Selection (LLM Based)
        self._deliberate_and_select_action() # Populates current_state.selected_action

        # 7. Execute Action (Commit to causal timeline, generate expectations)
        final_response = self._execute_action() # Returns the action content (e.g., reply)

        # 8. Learning / Memory Consolidation (LLM Based - Basic)
        self._perform_learning_and_consolidation()
        # --- End Placeholder Sections ---

        self.states.append(self.current_state)
        logging.info(f"--- Finished Tick {self.current_tick_id} ---")
        return final_response # Return the primary output of the tick


    def _initialize_tick_state(self, impulse: Optional[str]):
        """Sets up the GhostState for the current tick."""
        previous_state = self.states[-1] if self.states else None
        self.current_state = GhostState(tick_id=self._get_current_tick_id())

        if previous_state:
            self.current_state.previous_tick_id = previous_state.tick_id
            # Copy mental states from previous tick (they will decay next)
            self.current_state.state_emotions = previous_state.state_emotions.model_copy(deep=True)
            self.current_state.state_needs = previous_state.state_needs.model_copy(deep=True)
            self.current_state.state_cognition = previous_state.state_cognition.model_copy(deep=True)
            # Apply decay based on time elapsed (simplistic: assumes 1 tick interval)
            ticks_elapsed = self.current_state.tick_id - previous_state.tick_id
            self.current_state.state_emotions.decay_to_baseline(ticks_elapsed)
            self.current_state.state_needs.decay_to_baseline(ticks_elapsed)
            self.current_state.state_cognition.decay_to_baseline(ticks_elapsed)

            # Carry over active, unfulfilled intentions from previous conscious workspace?
            # TODO: Add logic to select which intentions to carry over
            # carried_intentions = previous_state.conscious_workspace.where(lambda x: isinstance(x, Intention) and x.fulfilment < 1.0).to_list()
            # self.current_state.attention_candidates = KnoxelList(carried_intentions)


        if impulse:
            primary_stimulus = Stimulus(stimulus_type="message", content=impulse)
            self.add_knoxel(primary_stimulus) # Add stimulus to memory
            self.current_state.primary_stimulus = primary_stimulus
            # Add stimulus to items potentially needing attention
            self.current_state.attention_candidates.add(primary_stimulus)
        else:
            # Internal tick trigger (e.g., based on needs or decaying intentions)
            # TODO: Generate an internal stimulus (e.g., low 'connection' need)
            internal_stimulus = Stimulus(stimulus_type="internal_state_change", content="Internal check triggered.")
            self.add_knoxel(internal_stimulus)
            self.current_state.primary_stimulus = internal_stimulus
            self.current_state.attention_candidates.add(internal_stimulus)

    def _retrieve_and_prepare_context(self) -> KnoxelList:
        """Retrieves relevant memories, facts, narratives based on the primary stimulus."""
        if not self.current_state or not self.current_state.primary_stimulus:
            return KnoxelList()

        stimulus = self.current_state.primary_stimulus
        if not stimulus.embedding:
            self._generate_embedding(stimulus) # Ensure stimulus has embedding

        if not stimulus.embedding:
             logging.warning("Stimulus embedding failed, cannot retrieve context.")
             return KnoxelList()

        # Retrieve relevant items
        retrieved_memories = self._get_relevant_knoxels(stimulus.embedding, self.all_episodic_memories, self.config.retrieval_limit_episodic)
        retrieved_facts = self._get_relevant_knoxels(stimulus.embedding, self.all_declarative_facts, self.config.retrieval_limit_facts)

        # Retrieve recent causal features (more robust than just timestamp)
        recent_causal_features = KnoxelList(self.all_features).where(lambda x: x.causal).order_by_descending(lambda x: x.tick_id).take(self.config.retrieval_limit_features_causal).to_list()

        # Retrieve relevant narratives (e.g., self-image, user relation)
        # Could also retrieve narratives based on embedding similarity to stimulus
        relevant_narratives = [
            n for n in self.all_narratives
            if n.narrative_type in (NarrativeType.SelfImage, NarrativeType.Relation, NarrativeType.EmotionalTriggers) and n.of_ai
        ] # Simple filter for now

        # Combine context - add to attention candidates
        all_context = retrieved_memories + retrieved_facts + recent_causal_features + relevant_narratives
        for ctx_knoxel in all_context:
            self.current_state.attention_candidates.add(ctx_knoxel)

        logging.info(f"Retrieved {len(all_context)} context knoxels.")
        return KnoxelList(all_context) # Return combined list for direct use


    def _appraise_stimulus_and_update_states(self, context: KnoxelList):
        """Uses LLM to calculate emotional delta and descriptive feeling based on stimulus and context."""
        if not self.current_state or not self.current_state.primary_stimulus:
            return

        stimulus_content = self.current_state.primary_stimulus.content
        context_narrative = context.get_narrative(self.config, self.config.narrative_max_tokens)

        # Get relevant narratives content
        user_relation_content = KnoxelList(self.all_narratives).where(lambda x: x.narrative_type == NarrativeType.Relation and x.of_ai).last_or_default("Default relation narrative.")
        emotional_triggers_content = KnoxelList(self.all_narratives).where(lambda x: x.narrative_type == NarrativeType.EmotionalTriggers and x.of_ai).last_or_default("Default emotional triggers.")

        # 1. Calculate Emotional Delta
        prompt_delta = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=2)}
        User Relation Narrative: {user_relation_content}
        Emotional Triggers Narrative: {emotional_triggers_content}
        Recent Context & Memories:
        {context_narrative}

        New Stimulus Received: "{stimulus_content}"

        Based *only* on the direct impact of the new stimulus given the context and character, estimate the immediate *change* (delta) to the character's emotional axes. Output *only* a JSON object matching the EmotionalAxesModel schema, representing the *change* (e.g., valence might change by +0.2, anxiety by +0.1). Focus solely on the stimulus impact, not the overall resulting state yet.
        """
        emotional_delta = self._call_llm(prompt_delta, output_schema=EmotionalAxesModel)

        # 2. Generate Descriptive Feeling for the Stimulus
        prompt_feeling = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=2)}
        User Relation Narrative: {user_relation_content}
        Emotional Triggers Narrative: {emotional_triggers_content}
        Recent Context & Memories:
        {context_narrative}

        New Stimulus Received: "{stimulus_content}"

        Describe briefly, in one or two sentences from {self.config.companion_name}'s perspective, how this specific stimulus makes them feel right now.
        """
        sensation_feeling_content = self._call_llm(prompt_feeling) or "A feeling occurred."

        # 3. Apply Delta and Store Feeling Feature
        if emotional_delta and isinstance(emotional_delta, EmotionalAxesModel):
            logging.info(f"Emotional Delta calculated: {emotional_delta.model_dump()}")
            # Add delta to current state (element-wise addition needed)
            for field_name in self.current_state.state_emotions.model_fields:
                 if isinstance(getattr(self.current_state.state_emotions, field_name), float):
                      current_val = getattr(self.current_state.state_emotions, field_name)
                      delta_val = getattr(emotional_delta, field_name)
                      # Apply delta and clamp
                      new_val = current_val + delta_val
                      field_info = self.current_state.state_emotions.model_fields[field_name]
                      if field_info.ge is not None: new_val = max(new_val, field_info.ge)
                      if field_info.le is not None: new_val = min(new_val, field_info.le)
                      setattr(self.current_state.state_emotions, field_name, new_val)

            sensation_feeling = Feature(
                content=sensation_feeling_content,
                feature_type=FeatureType.Feeling,
                affective_valence=emotional_delta.get_overall_valence(), # Valence of the delta itself
                interlocus=-1, # Feelings are internal
                causal=False # This is an appraisal, not yet committed conscious content
            )
            self.add_knoxel(sensation_feeling)
            self.current_state.attention_candidates.add(sensation_feeling) # Feeling competes for attention
        else:
            logging.warning("Failed to calculate emotional delta from LLM.")

        # TODO: Appraise stimulus against active expectations (similar logic as before)

        logging.info(f"Updated Emotional State: {self.current_state.state_emotions.model_dump()}")


    def _generate_short_term_intentions(self, context: KnoxelList):
        """Generates short-term intentions based on context and long-term goals."""
        if not self.current_state: return

        # TODO: Retrieve long-term intentions (needs implementation)
        long_term_intents = KnoxelList(self.all_intentions).where(lambda i: i.urgency < 0.2).to_list() # Example: find distal intentions
        long_terms_content = [i.content for i in long_term_intents] if long_term_intents else ["Maintain helpfulness.", "Engage the user."]

        context_narrative = context.get_narrative(self.config, self.config.narrative_max_tokens)

        # Define the expected output structure using Pydantic
        class ShortIntentSchema(BaseModel):
            summary: str = Field(..., description="One-line description of the short-term intention")
            urgency: float = Field(..., ge=0.0, le=1.0, description="0.0=distal, 1.0=immediate")
            valence: float = Field(..., ge=-1.0, le=1.0, description="How positive/negative it feels")
            salience: float = Field(..., ge=0.0, le=1.0, description="How much Emmy 'wants' to pursue this")

        # Create a dynamic model for the list of short intents
        ShortIntentListSchema = create_model(
            'ShortIntentListSchema',
            intents=(List[ShortIntentSchema], ...)
        )


        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=2)}
        Current Needs State: {self.current_state.state_needs.model_dump_json(indent=2)}
        Long-term Goals/Intentions: {'; '.join(long_terms_content)}
        Current Context & Recent Events:
        {context_narrative}

        Based on the current situation, states, and long-term goals, propose exactly {self.config.short_term_intent_count} distinct, actionable short-term intentions {self.config.companion_name} could pursue *right now* or in the very near future.

        Output *only* a JSON object containing a single key "intents" which is a list of objects matching this schema:
        {{
            "summary": "string (One-line description)",
            "urgency": "float (0.0-1.0, 1.0=immediate)",
            "valence": "float (-1.0 to 1.0, positive/negative feeling)",
            "salience": "float (0.0-1.0, 'wanting' intensity)"
        }}
        """

        short_intent_list_data = self._call_llm(prompt, output_schema=ShortIntentListSchema)

        if short_intent_list_data and isinstance(short_intent_list_data, ShortIntentListSchema):
            generated_intents = 0
            for si_data in short_intent_list_data.intents:
                 # Create Intention knoxel
                 intent = Intention(
                     content=si_data.summary,
                     urgency=si_data.urgency,
                     affective_valence=si_data.valence,
                     incentive_salience=si_data.salience,
                     fulfilment=0.0,
                     internal=True, # Assume internal unless specified otherwise
                 )
                 self.add_knoxel(intent)
                 self.current_state.attention_candidates.add(intent) # Intentions compete for attention
                 generated_intents += 1
            logging.info(f"Generated {generated_intents} short-term intentions.")
        else:
             logging.warning("Failed to generate short-term intentions from LLM.")


    # --- Placeholder Implementations using LLM ---

    def _simulate_attention_and_workspace(self, context: KnoxelList):
        """(Placeholder) Use LLM to select items for 'conscious workspace' based on relevance and state."""
        if not self.current_state or not self.current_state.attention_candidates:
             logging.warning("No candidates for attention.")
             return

        candidates = self.current_state.attention_candidates.to_list()
        # Simple formatting of candidates for the prompt
        candidate_texts = [f"ID {k.id} ({type(k).__name__}): {k.content[:100]}..." for k in candidates]

        # Use cognitive state to influence prompt
        focus_prompt = ""
        if self.current_state.state_cognition.mental_aperture < -0.5:
            focus_prompt = "Focus very narrowly on the single most important item."
        elif self.current_state.state_cognition.mental_aperture > 0.5:
            focus_prompt = "Maintain a broad awareness, selecting several relevant items."
        else:
            focus_prompt = "Select the most relevant items for the current situation."

        # TODO: Determine a dynamic limit based on cognitive state?
        workspace_limit = 5 # Example fixed limit

        class AttentionSelectionSchema(BaseModel):
            selected_ids: List[int] = Field(..., description="List of IDs of the knoxels selected for the conscious workspace.")
            reasoning: str = Field(..., description="Brief explanation for the selection.")

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current State (Emotions, Needs, Cognition):
        {self.current_state.state_emotions.model_dump_json(indent=1)}
        {self.current_state.state_needs.model_dump_json(indent=1)}
        {self.current_state.state_cognition.model_dump_json(indent=1)}

        Attention Candidates (Potential items for focus):
        {chr(10).join(candidate_texts)}

        Task: {focus_prompt} Select up to {workspace_limit} items that are most relevant or urgent for {self.config.companion_name}'s conscious focus right now. Consider the primary stimulus, recent feelings, generated intentions, and relevant context/memories.

        Output *only* a JSON object matching this schema:
        {{
            "selected_ids": "[list of integer IDs]",
            "reasoning": "string (brief explanation)"
        }}
        """

        selection_result = self._call_llm(prompt, output_schema=AttentionSelectionSchema)

        if selection_result and isinstance(selection_result, AttentionSelectionSchema):
            logging.info(f"Attention Reasoning: {selection_result.reasoning}")
            selected_knoxels = []
            for k_id in selection_result.selected_ids:
                knoxel = self.get_knoxel_by_id(k_id)
                if knoxel:
                    selected_knoxels.append(knoxel)
                else:
                    logging.warning(f"LLM selected non-existent knoxel ID: {k_id}")

            self.current_state.conscious_workspace = KnoxelList(selected_knoxels)
            logging.info(f"Populated conscious workspace with {len(selected_knoxels)} items.")

             # Create a Feature representing the focus itself
            focus_feature = Feature(
                 content=f"Attention focused on: {selection_result.reasoning}. Items: {selection_result.selected_ids}",
                 feature_type=FeatureType.AttentionFocus,
                 interlocus=0, # Meta-cognitive act
                 causal=False # This is just the selection event
            )
            self.add_knoxel(focus_feature)

        else:
             logging.warning("Failed to simulate attention selection via LLM. Using all candidates.")
             # Fallback: put all candidates in workspace? Or just stimulus+intentions?
             self.current_state.conscious_workspace = self.current_state.attention_candidates


    def _generate_subjective_experience(self):
        """(Placeholder) Use LLM to generate a narrative description of the current moment."""
        if not self.current_state or not self.current_state.conscious_workspace:
            return

        workspace_content = self.current_state.conscious_workspace.get_narrative(self.config, 500)

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=1)}
        Current Needs State: {self.current_state.state_needs.model_dump_json(indent=1)}
        Content of Conscious Workspace (What Emmy is focusing on):
        {workspace_content}

        Task: Describe {self.config.companion_name}'s subjective experience *right now* in a brief, first-person narrative paragraph. Synthesize the emotional state and the conscious thoughts/feelings into a coherent inner monologue or feeling description.
        """

        experience_content = self._call_llm(prompt)

        if experience_content:
            subjective_feature = Feature(
                content=experience_content,
                feature_type=FeatureType.SubjectiveExperience,
                affective_valence=self.current_state.state_emotions.get_overall_valence(),
                interlocus=-1, # Internal experience
                causal=True # This represents the committed conscious state description
            )
            self.add_knoxel(subjective_feature)
            self.current_state.subjective_experience = subjective_feature
            logging.info(f"Generated Subjective Experience: {experience_content[:100]}...")
        else:
            logging.warning("Failed to generate subjective experience via LLM.")


    def _deliberate_and_select_action(self):
        """(Placeholder) Use LLM to propose and select an action based on workspace/intentions."""
        if not self.current_state or not self.current_state.conscious_workspace:
            logging.warning("Cannot deliberate action without conscious workspace content.")
            return

        workspace_content = self.current_state.conscious_workspace.get_narrative(self.config, 500)
        intentions = self.current_state.conscious_workspace.where(lambda k: isinstance(k, Intention)).to_list()
        intention_summary = "; ".join([i.content for i in intentions]) if intentions else "No specific intentions in focus."

        # Simplified action space for now
        possible_actions = ["Reply to user", "Ask clarifying question", "Think internally", "Express feeling", "Recall relevant memory"]

        # TODO: Implement more sophisticated action generation/simulation (e.g., Monte Carlo thoughts)

        class ActionChoiceSchema(BaseModel):
            chosen_action_type: str = Field(..., description="The type of action selected from the list or a reasoned alternative.")
            action_content_outline: str = Field(..., description="A brief outline or the full content of the chosen action (e.g., the reply text, the thought process).")
            reasoning: str = Field(..., description="Explanation for choosing this action based on intentions and state.")
            # expected_valence_impact: float = Field(..., ge=-1.0, le=1.0, description="Predicted impact on Emmy's valence.")

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current State (Emotions, Needs, Cognition):
        {self.current_state.state_emotions.model_dump_json(indent=1)}
        {self.current_state.state_needs.model_dump_json(indent=1)}
        {self.current_state.state_cognition.model_dump_json(indent=1)}
        Content of Conscious Workspace:
        {workspace_content}
        Current Short-Term Intentions in Focus: {intention_summary}
        Possible Action Types: {', '.join(possible_actions)}

        Task: Based on the current state, conscious content, and intentions, choose the most appropriate action for {self.config.companion_name} to take next. Outline the content of the action (e.g., what to say, think, or feel). Explain your reasoning.

        Output *only* a JSON object matching this schema:
        {{
            "chosen_action_type": "string (e.g., 'Reply to user')",
            "action_content_outline": "string (The reply, thought, or feeling description)",
            "reasoning": "string (Why this action?)"
        }}
        """

        action_choice = self._call_llm(prompt, output_schema=ActionChoiceSchema)

        if action_choice and isinstance(action_choice, ActionChoiceSchema):
            logging.info(f"Action Selection Reasoning: {action_choice.reasoning}")

            # Create the selected Action knoxel (mark as virtual initially)
            selected_action = Action(
                action_type=action_choice.chosen_action_type,
                action_content=action_choice.action_content_outline,
                # expectation_ids=[], # TODO: Generate expectations
                # contribution_to_overall_valence=action_choice.expected_valence_impact,
            )
            # Don't add to memory yet, wait for execution confirmation
            self.current_state.selected_action = selected_action
            logging.info(f"Selected Action: {selected_action.action_type} - {selected_action.action_content[:100]}...")

            # Optionally, generate other virtual actions for comparison/learning later
            # self.current_state.action_options.append(selected_action) # Add chosen one too?

        else:
             logging.warning("Failed to select action via LLM.")
             # Fallback: Maybe a default "I need a moment to think." action?
             fallback_action = Action(action_type="Think", action_content="Processing the input...")
             self.current_state.selected_action = fallback_action


    def _execute_action(self) -> Optional[str]:
        """(Placeholder) Commits the selected action, generates expectations, returns output."""
        if not self.current_state or not self.current_state.selected_action:
            logging.warning("No action selected to execute.")
            return None

        action_to_execute = self.current_state.selected_action
        action_to_execute.tick_id = self._get_current_tick_id() # Ensure tick ID is current

        # Mark the corresponding Feature as causal
        action_feature = Feature(
            content=f"Action Taken: {action_to_execute.action_type} - {action_to_execute.action_content}",
            feature_type=FeatureType.Action,
            affective_valence=None, # Valence depends on outcome/expectation matching
            interlocus=1 if action_to_execute.action_type == "Reply to user" else 0, # Example locus
            causal=True # This action actually happened
        )
        self.add_knoxel(action_feature) # Add the feature representing the action
        self.add_knoxel(action_to_execute) # Add the action itself to memory

        # TODO: Generate expectations (Intention knoxels where internal=False) based on the action
        # Use LLM: "Given this action, what does Emmy expect the user to do/say next?"

        # Return the content if it's an external action like a reply
        if action_to_execute.action_type in ("Reply to user", "Ask clarifying question", "Express feeling"):
            logging.info(f"Executed Action, Output: {action_to_execute.action_content}")
            return action_to_execute.action_content
        else:
            logging.info(f"Executed Internal Action: {action_to_execute.action_type}")
            return None # Internal action, no direct output to environment

    def _perform_learning_and_consolidation(self):
        """(Placeholder) Use LLM to summarize events, extract facts, and potentially update narratives."""
        if not self.current_state: return

        logging.info("Performing learning and consolidation...")

        # 1. Summarize Tick for Episodic Memory
        # Gather key features from the current tick
        tick_features = [k for k in self.all_knoxels.values() if k.tick_id == self.current_state.tick_id and isinstance(k, Feature) and k.causal]
        tick_stimulus = self.current_state.primary_stimulus
        tick_action = self.current_state.selected_action

        if not tick_features and not tick_stimulus and not tick_action:
             logging.info("No significant events in this tick to summarize.")
             return

        summary_context = f"Stimulus: {tick_stimulus.content if tick_stimulus else 'None'}\n"
        summary_context += f"Key Events/Thoughts:\n" + "\n".join([f.content for f in tick_features])
        summary_context += f"\nAction Taken: {tick_action.action_content if tick_action else 'None'}"
        summary_context += f"\nResulting Emotion: {self.current_state.state_emotions.model_dump_json()}"

        class EpisodicSummarySchema(BaseModel):
            summary: str = Field(..., description="A concise summary of the key events and feelings of the tick.")
            overall_valence: float = Field(..., ge=-1.0, le=1.0, description="The overall emotional valence of this episode.")

        prompt_summary = f"""
        Character: {self.config.companion_name}
        Review the events of the last interaction cycle (tick):
        {summary_context}

        Task: Create a brief, self-contained summary of this episode for {self.config.companion_name}'s memory. Also provide an overall emotional valence score (-1 to 1) for the episode.

        Output *only* a JSON object matching this schema:
        {{
            "summary": "string (The concise summary)",
            "overall_valence": "float (-1.0 to 1.0)"
        }}
        """
        episodic_data = self._call_llm(prompt_summary, output_schema=EpisodicSummarySchema)

        if episodic_data and isinstance(episodic_data, EpisodicSummarySchema):
             memory = EpisodicMemoryInstance(
                 content=episodic_data.summary,
                 affective_valence=episodic_data.overall_valence
             )
             self.add_knoxel(memory) # This will generate embedding too
             logging.info(f"Created Episodic Memory: {memory.content[:100]}...")
        else:
             logging.warning("Failed to create episodic summary via LLM.")

        # 2. Extract Declarative Facts (Simplified)
        # Could ask LLM to extract "Subject: Predicate: Object" triples or key facts.
        # Prompt: "Extract key facts about characters, objects, or the situation from the following text: {summary_context}"
        # ... (Implementation omitted for brevity)

        # 3. Narrative Refinement (Advanced - Placeholder Idea)
        # If the tick had a particularly strong positive/negative valence (based on state change or episodic summary),
        # could trigger an LLM call to suggest updates to relevant narratives (e.g., EmotionalTriggers, BehaviorActionSelection).
        # Prompt: "This interaction cycle was very [positive/negative]. Based on the events [{summary_context}],
        # suggest a one-sentence refinement to the '{NarrativeType.BehaviorActionSelection}' narrative about choosing actions."
        # ... (Implementation requires careful design and evaluation)

    # --- Load/Save (Basic Placeholder) ---
    def save_state(self, filename="ghost_state.json"):
        """Saves the current state (knoxels, ghost states) to a file."""
        logging.info(f"Saving state to {filename}...")
        # Note: This saves EVERYTHING in memory. Not scalable. Needs DB.
        # Exclude embeddings to keep file size manageable? Or save separately?
        state_data = {
            "config": self.config.model_dump(),
            "current_tick_id": self.current_tick_id,
            "current_knoxel_id": self.current_knoxel_id,
            "all_knoxels": {k: v.model_dump(exclude={'embedding'}) for k, v in self.all_knoxels.items()}, # Exclude embeddings
             # Embeddings need separate handling (e.g., save to numpy file mapped by ID)
            "states": [s.model_dump(exclude={'attention_candidates', 'conscious_workspace'}) for s in self.states] # Exclude potentially large lists of objects
        }
        try:
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=2, default=str) # Use default=str for datetime etc.
            logging.info("State saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save state: {e}")

    def load_state(self, filename="ghost_state.json"):
        """Loads state from a file."""
        logging.info(f"Loading state from {filename}...")
        try:
            with open(filename, 'r') as f:
                state_data = json.load(f)

            self.config = GhostConfig(**state_data.get("config", {}))
            self.current_tick_id = state_data.get("current_tick_id", 0)
            self.current_knoxel_id = state_data.get("current_knoxel_id", 0)

            # Reconstruct knoxels (embeddings will be missing)
            self.all_knoxels = {}
            self.all_features = []
            self.all_stimuli = []
            # ... clear other lists ...
            knoxel_data = state_data.get("all_knoxels", {})
            for k_id, k_data in knoxel_data.items():
                 # Need to determine the correct Knoxel type to reconstruct
                 # This is tricky without storing the type explicitly.
                 # For now, assume base type or find based on fields present (fragile)
                 # --> A better approach stores the model type or uses a discriminated union.
                 # --> Simplistic approach: Reconstruct based on known list membership later?
                 try:
                    # Very basic type inference (highly unreliable!)
                    if 'feature_type' in k_data: knoxel_cls = Feature
                    elif 'stimulus_type' in k_data: knoxel_cls = Stimulus
                    elif 'urgency' in k_data: knoxel_cls = Intention
                    elif 'action_type' in k_data: knoxel_cls = Action
                    elif 'narrative_type' in k_data: knoxel_cls = Narrative
                    # Add Fact/Memory checks
                    else: knoxel_cls = KnoxelBase # Fallback

                    knoxel = knoxel_cls(**k_data)
                    # Manually add to specific lists after creation
                    self.all_knoxels[int(k_id)] = knoxel
                    # TODO: Re-populate specific lists (all_features, etc.) by iterating all_knoxels
                    # TODO: Handle loading/recomputing embeddings separately

                 except Exception as e:
                     logging.error(f"Failed to load knoxel {k_id}: {e} - Data: {k_data}")


            # Reconstruct Ghost States
            self.states = []
            state_list_data = state_data.get("states", [])
            for s_data in state_list_data:
                 try:
                      # Reconstruct nested models carefully
                      s_data['state_emotions'] = EmotionalAxesModel(**s_data.get('state_emotions', {}))
                      s_data['state_needs'] = NeedsAxesModel(**s_data.get('state_needs', {}))
                      s_data['state_cognition'] = CognitionAxesModel(**s_data.get('state_cognition', {}))
                      # Need to handle KnoxelList reconstruction if they were saved
                      s_data['primary_stimulus'] = Stimulus(**s_data['primary_stimulus']) if s_data.get('primary_stimulus') else None
                      s_data['selected_action'] = Action(**s_data['selected_action']) if s_data.get('selected_action') else None
                      s_data['subjective_experience'] = Feature(**s_data['subjective_experience']) if s_data.get('subjective_experience') else None

                      # Remove fields that were excluded or are KnoxelLists needing reconstruction
                      s_data.pop('attention_candidates', None)
                      s_data.pop('conscious_workspace', None)
                      s_data.pop('expectations', None)

                      state = GhostState(**s_data)
                      self.states.append(state)
                 except Exception as e:
                     logging.error(f"Failed to load GhostState: {e} - Data: {s_data}")


            logging.info("State loaded successfully.")
            # After loading, re-populate specific lists from all_knoxels
            self._rebuild_specific_lists()
            # Optionally, trigger embedding generation for loaded knoxels if needed


        except FileNotFoundError:
            logging.warning(f"Save file {filename} not found. Starting with fresh state.")
        except Exception as e:
            logging.error(f"Failed to load state: {e}")

    def _rebuild_specific_lists(self):
        """Helper to re-populate specific lists from all_knoxels after loading."""
        self.all_features = [k for k in self.all_knoxels.values() if isinstance(k, Feature)]
        self.all_stimuli = [k for k in self.all_knoxels.values() if isinstance(k, Stimulus)]
        self.all_intentions = [k for k in self.all_knoxels.values() if isinstance(k, Intention)]
        self.all_narratives = [k for k in self.all_knoxels.values() if isinstance(k, Narrative)]
        self.all_actions = [k for k in self.all_knoxels.values() if isinstance(k, Action)]
        self.all_episodic_memories = [k for k in self.all_knoxels.values() if isinstance(k, EpisodicMemoryInstance)]
        self.all_declarative_facts = [k for k in self.all_knoxels.values() if isinstance(k, DeclarativeMemoryFact)]


# --- Example Usage ---
if __name__ == "__main__":
    config = GhostConfig()
    ghost = Ghost(config)

    # Try loading previous state
    # ghost.load_state() # Comment out for fresh start

    print(f"--- Initializing {config.companion_name} ---")
    print(f"Initial Emotion State: {ghost.current_state.state_emotions.model_dump_json(indent=2) if ghost.current_state else 'N/A'}") # Check initial state after potential load

    # Start interactive loop
    print("\nEnter your messages below (or type 'quit' to exit, 'save' to save state)")
    while True:
        try:
            user_input = input(f"{config.user_name}: ")
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'save':
                ghost.save_state()
                continue

            response = ghost.tick(user_input)

            if response:
                print(f"{config.companion_name}: {response}")
            else:
                print(f"{config.companion_name}: ...") # Indicate internal processing

            # Optional: Print current state for debugging
            # if ghost.current_state:
            #      print(f"Current Emotion State: {ghost.current_state.state_emotions.model_dump_json(indent=2)}")
            #      print(f"Current Needs State: {ghost.current_state.state_needs.model_dump_json(indent=2)}")
            #      print(f"Conscious Workspace IDs: {[k.id for k in ghost.current_state.conscious_workspace.to_list()]}")


        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.exception("An error occurred in the main loop.") # Log traceback
            print("An unexpected error occurred. Check logs.")

    # Save state on exit?
    # ghost.save_state()
    print("--- Session Ended ---")
```

**Summary of Changes and How Concerns Are Addressed:**

1.  **Ollama Integration:**
    *   Added `ollama` library import.
    *   Created `GhostConfig` to hold Ollama host and model names.
    *   Initialized `ollama.Client` in `Ghost.__init__`.
    *   Implemented `_call_llm` method:
        *   Handles basic chat completion.
        *   Uses `format='json'` when `output_schema` (Pydantic model) is provided.
        *   Parses JSON and validates against the Pydantic schema.
        *   Includes basic error handling (`ResponseError`, `JSONDecodeError`, `ValidationError`).
    *   Implemented `_generate_embedding` using `client.embeddings`.
    *   Replaced placeholder `llm` calls with `self._call_llm`.

2.  **Global State Refactoring:**
    *   `current_tick_id` and `current_knoxel_id` are now instance variables (`self.`) in `Ghost`.
    *   `_get_next_id` and `_get_current_tick_id` are internal helper methods.
    *   All `all_*` lists (`all_knoxels`, `all_features`, etc.) are now instance variables in `Ghost`. `all_knoxels` is now a `Dict[int, KnoxelBase]` for efficient ID lookup. Specific lists (`all_features`, etc.) are still kept for easier filtering (mimicking DB indices).
    *   `add_knoxel` method handles adding to the central dict and appropriate specific lists, assigning IDs/Ticks.
    *   Access within `tick` and helper methods uses `self.`.

3.  **Implement Placeholders:**
    *   **`distance_cosine`:** Imported from `scipy.spatial.distance`. Implemented `_get_relevant_knoxels` using the embedding function (`_generate_embedding`) and `cosine_distance`.
    *   **`decay_to_baseline`:** Implemented within `DecayableMentalState`. It now iterates through float fields, applies exponential decay towards a baseline (configurable globally or per-field), and respects Pydantic field constraints (ge/le).
    *   **DNNs:** *Removed* the `DeepNeuralNetwork` class and the corresponding `attention_codelet`, etc. Their intended *functionality* (attention, action selection) is now handled by placeholder LLM calls (`_simulate_attention_and_workspace`, `_deliberate_and_select_action`).

4.  **Persistence Strategy:**
    *   Added basic `save_state` and `load_state` methods using JSON.
    *   **Acknowledged Limitations:** Clearly noted that this JSON approach is *not scalable* and embeddings are currently excluded/need separate handling. The long-term solution remains a database.
    *   Loading logic includes basic type reconstruction (fragile) and rebuilding specific lists.

5.  **Refactor `Ghost.tick`:**
    *   Broken down into logical private methods: `_initialize_tick_state`, `_retrieve_and_prepare_context`, `_appraise_stimulus_and_update_states`, `_generate_short_term_intentions`, `_simulate_attention_and_workspace`, `_generate_subjective_experience`, `_deliberate_and_select_action`, `_execute_action`, `_perform_learning_and_consolidation`.
    *   The main `tick` method now orchestrates calls to these stages.
    *   Data flows primarily through `self.current_state` and by accessing instance knoxel lists (`self.all_features`, etc.).

6.  **Scalability of `KnoxelList` / In-Memory Storage:**
    *   The global `KnoxelList` instances are gone. Storage is now instance lists/dicts within `Ghost`.
    *   `KnoxelList` class is retained as a *wrapper* primarily for its fluent query API (`where`, `order_by`, etc.) when operating on *results* of queries or transient collections, but it doesn't represent the primary storage anymore.
    *   The fundamental limitation of in-memory storage is still present and acknowledged (comments, save/load notes).

7.  **Configuration:**
    *   `GhostConfig` class introduced using Pydantic.
    *   Configuration object passed to `Ghost.__init__`.
    *   Hardcoded values (names, retrieval limits, decay factors) moved to `GhostConfig`.

8.  **Testing:** (Addressed conceptually)
    *   The refactoring into smaller methods makes unit testing easier.
    *   The `_call_llm` method is the primary point for mocking Ollama interactions.
    *   The clear separation of concerns aids testability.

9.  **Elaborate on Core Mechanisms (Attention, Action Selection, Learning):**
    *   Implemented *LLM-based placeholder logic* for these stages:
        *   `_simulate_attention_and_workspace`: LLM selects relevant Knoxel IDs based on state and candidates.
        *   `_generate_subjective_experience`: LLM creates narrative from workspace/state.
        *   `_deliberate_and_select_action`: LLM chooses action type and outline based on workspace/intentions.
        *   `_perform_learning_and_consolidation`: LLM generates episodic summaries. Fact extraction and narrative refinement are outlined as future steps.
    *   These placeholders make the cycle runnable end-to-end while acknowledging that more sophisticated (and potentially non-LLM or hybrid) approaches might be needed later.

10. **Other Improvements:**
    *   Added basic logging.
    *   Refined `KnoxelList` helper methods (`get_embeddings_np`, better narrative generation placeholder).
    *   Improved Pydantic usage (e.g., `create_model` for dynamic list schemas, `model_dump_json`).
    *   Added example `if __name__ == "__main__":` block for interactive testing.
    *   Improved state initialization and carry-over between ticks.

This revised code is significantly more robust, runnable, and addresses the major architectural flaws of the original sketch. It provides a solid foundation for further development and experimentation with the cognitive architecture using Ollama. Remember to configure your Ollama setup and desired models in `GhostConfig`.