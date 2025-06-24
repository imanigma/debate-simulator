import streamlit as st
from openai import OpenAI
from typing import List, Dict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from typing import TypedDict
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Enhanced Agent Profile System (inspired by AgentSociety)
class PersonalityTrait(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

class SocialRole(Enum):
    COMMUNITY_LEADER = "community_leader"
    BUSINESS_OWNER = "business_owner"
    RESIDENT = "resident"
    POLICY_MAKER = "policy_maker"
    ACTIVIST = "activist"
    ACADEMIC = "academic"

@dataclass
class AgentProfile:
    """Enhanced agent profile with psychological and social attributes"""
    name: str
    age: int
    social_role: SocialRole
    background: str
    core_values: List[str]
    personality_traits: Dict[PersonalityTrait, float]  # 0-1 scale
    experience_level: float  # 0-1 scale
    emotional_state: Dict[str, float]  # current emotional state
    memory_strength: float = 0.8
    persuasion_style: str = "logical"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DebateMessage:
    """Enhanced message structure with metadata"""
    speaker_id: str
    speaker_name: str
    content: str
    timestamp: datetime
    round_number: int
    message_type: str = "argument"  # argument, rebuttal, question, clarification
    emotional_tone: Dict[str, float] = None
    cited_sources: List[str] = None
    
    def __post_init__(self):
        if self.emotional_tone is None:
            self.emotional_tone = {}
        if self.cited_sources is None:
            self.cited_sources = []

class AgentMemorySystem:
    """Enhanced memory system for agents"""
    def __init__(self, agent_id: str, memory_strength: float = 0.8):
        self.agent_id = agent_id
        self.memory_strength = memory_strength
        self.short_term_memory = ConversationBufferWindowMemory(k=5)
        self.long_term_memory = []
        self.key_points = []
        self.opponent_model = {}  # Model of opponent's arguments and weaknesses
    
    def add_message(self, message: DebateMessage):
        """Add message to memory with importance weighting"""
        self.short_term_memory.chat_memory.add_message(
            HumanMessage(content=f"{message.speaker_name}: {message.content}")
        )
        
        # Store important messages in long-term memory
        importance_score = self._calculate_importance(message)
        if importance_score > 0.7:
            self.long_term_memory.append({
                'message': message,
                'importance': importance_score,
                'retrieval_count': 0
            })
    
    def _calculate_importance(self, message: DebateMessage) -> float:
        """Calculate importance score for memory storage"""
        score = 0.5
        
        # Boost importance for rebuttals and strong statements
        if message.message_type == "rebuttal":
            score += 0.2
        if len(message.content) > 200:  # Detailed arguments
            score += 0.1
        if message.cited_sources:
            score += 0.2
            
        return min(score, 1.0)
    
    def retrieve_relevant_memories(self, context: str) -> List[Dict]:
        """Retrieve relevant memories based on context"""
        relevant = []
        for memory in self.long_term_memory:
            # Simple relevance scoring (could be enhanced with embeddings)
            if any(word in context.lower() for word in memory['message'].content.lower().split()):
                relevant.append(memory)
                memory['retrieval_count'] += 1
        
        return sorted(relevant, key=lambda x: x['importance'], reverse=True)[:3]

class EnhancedDebateAgent:
    """Enhanced agent with personality, memory, and strategic reasoning"""
    
    def __init__(self, profile: AgentProfile, api_key: str):
        self.profile = profile
        self.agent_id = str(uuid.uuid4())
        self.memory = AgentMemorySystem(self.agent_id, profile.memory_strength)
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Using OpenAI o1-mini as requested
            openai_api_key=api_key,
            max_tokens=None
        )
        self.debate_strategy = self._develop_strategy()
    
    def _develop_strategy(self) -> Dict[str, Any]:
        """Develop debate strategy based on personality and role"""
        strategy = {
            'primary_approach': self._get_primary_approach(),
            'fallback_tactics': self._get_fallback_tactics(),
            'emotional_regulation': self.profile.personality_traits.get(PersonalityTrait.NEUROTICISM, 0.5),
            'collaboration_tendency': self.profile.personality_traits.get(PersonalityTrait.AGREEABLENESS, 0.5)
        }
        return strategy
    
    def _get_primary_approach(self) -> str:
        """Determine primary debate approach based on profile"""
        if self.profile.social_role == SocialRole.ACADEMIC:
            return "evidence_based"
        elif self.profile.social_role == SocialRole.ACTIVIST:
            return "passionate_advocacy"
        elif self.profile.social_role == SocialRole.BUSINESS_OWNER:
            return "practical_economics"
        else:
            return "community_focused"
    
    def _get_fallback_tactics(self) -> List[str]:
        """Get fallback tactics when primary approach isn't working"""
        tactics = ["reframe_issue", "find_common_ground"]
        
        if self.profile.personality_traits.get(PersonalityTrait.EXTRAVERSION, 0.5) > 0.7:
            tactics.append("emotional_appeal")
        if self.profile.personality_traits.get(PersonalityTrait.OPENNESS, 0.5) > 0.7:
            tactics.append("creative_solutions")
            
        return tactics
    
    def generate_response(self, debate_context: Dict, topic: str, round_num: int) -> DebateMessage:
        """Generate contextual response based on agent profile and memory"""
        
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve_relevant_memories(topic)
        
        # Build context-aware prompt
        system_prompt = self._build_system_prompt()
        context_prompt = self._build_context_prompt(debate_context, relevant_memories, topic, round_num)
        
        try:
            # Generate response using LangChain
            messages = [
                HumanMessage(content=f"System: {system_prompt}\n\nContext: {context_prompt}")
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Create structured message
            message = DebateMessage(
                speaker_id=self.agent_id,
                speaker_name=self.profile.name,
                content=content,
                timestamp=datetime.now(),
                round_number=round_num,
                message_type=self._determine_message_type(content, round_num),
                emotional_tone=self._analyze_emotional_tone(content)
            )
            
            # Add to memory
            self.memory.add_message(message)
            
            return message
            
        except Exception as e:
            # Fallback response
            return DebateMessage(
                speaker_id=self.agent_id,
                speaker_name=self.profile.name,
                content=f"I apologize, but I'm having difficulty formulating my response. Error: {str(e)}",
                timestamp=datetime.now(),
                round_number=round_num
            )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on agent profile"""
        personality_desc = self._describe_personality()
        
        return f"""You are {self.profile.name}, a {self.profile.age}-year-old {self.profile.social_role.value.replace('_', ' ')}.

Background: {self.profile.background}

Core Values: {', '.join(self.profile.core_values)}

Personality: {personality_desc}

Debate Style: Your primary approach is {self.debate_strategy['primary_approach']}. 
You tend to be {'collaborative' if self.debate_strategy['collaboration_tendency'] > 0.6 else 'competitive'} in debates.

Instructions:
- Stay true to your values and background
- Draw from your professional/personal experience
- Maintain consistency with your personality
- Engage constructively while advocating for your position
- Keep responses under 300 words
"""
    
    def _describe_personality(self) -> str:
        """Generate personality description from traits"""
        traits = []
        for trait, score in self.profile.personality_traits.items():
            if score > 0.7:
                traits.append(f"highly {trait.value}")
            elif score > 0.5:
                traits.append(f"moderately {trait.value}")
        
        return ", ".join(traits) if traits else "balanced personality"
    
    def _build_context_prompt(self, debate_context: Dict, memories: List, topic: str, round_num: int) -> str:
        """Build context-aware prompt"""
        context = f"Debate Topic: {topic}\nRound: {round_num}\n\n"
        
        if debate_context.get('previous_messages'):
            context += "Recent Discussion:\n"
            for msg in debate_context['previous_messages'][-3:]:  # Last 3 messages
                context += f"{msg['speaker']}: {msg['content']}\n"
        
        if memories:
            context += "\nRelevant Previous Points:\n"
            for memory in memories:
                context += f"- {memory['message'].content[:100]}...\n"
        
        context += f"\nPlease provide your {self.debate_strategy['primary_approach']} response to this debate."
        
        return context
    
    def _determine_message_type(self, content: str, round_num: int) -> str:
        """Determine message type based on content and context"""
        content_lower = content.lower()
        
        if round_num == 1:
            return "opening_statement"
        elif any(word in content_lower for word in ["however", "but", "disagree", "wrong"]):
            return "rebuttal"
        elif "?" in content:
            return "question"
        else:
            return "argument"
    
    def _analyze_emotional_tone(self, content: str) -> Dict[str, float]:
        """Simple emotional tone analysis"""
        content_lower = content.lower()
        
        emotions = {
            'confident': len([w for w in ['certain', 'clearly', 'obviously', 'definitely'] if w in content_lower]) * 0.2,
            'passionate': len([w for w in ['strongly', 'deeply', 'passionate', 'urgent'] if w in content_lower]) * 0.2,
            'analytical': len([w for w in ['data', 'evidence', 'research', 'study'] if w in content_lower]) * 0.2,
            'collaborative': len([w for w in ['together', 'common', 'shared', 'unite'] if w in content_lower]) * 0.2
        }
        
        return {k: min(v, 1.0) for k, v in emotions.items()}

# Enhanced Debate State
class EnhancedDebateState(TypedDict):
    conversation: List[Dict]
    round: int
    max_rounds: int
    topic: str
    agent_a: EnhancedDebateAgent
    agent_b: EnhancedDebateAgent
    verdict: Optional[str]
    debate_metrics: Dict[str, Any]

# Initialize session state with enhanced features
def initialize_session_state():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'judgement' not in st.session_state:
        st.session_state.judgement = ""
    if 'debate_metrics' not in st.session_state:
        st.session_state.debate_metrics = {}
    if 'agent_profiles' not in st.session_state:
        st.session_state.agent_profiles = create_default_profiles()

def create_default_profiles() -> Dict[str, AgentProfile]:
    """Create default agent profiles"""
    return {
        'community_builder': AgentProfile(
            name="Maya Rodriguez",
            age=35,
            social_role=SocialRole.COMMUNITY_LEADER,
            background="Urban planner with 10 years experience in affordable housing advocacy",
            core_values=["social equity", "community empowerment", "sustainable development"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.8,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
                PersonalityTrait.EXTRAVERSION: 0.7,
                PersonalityTrait.AGREEABLENESS: 0.8,
                PersonalityTrait.NEUROTICISM: 0.3
            },
            experience_level=0.8,
            emotional_state={"motivated": 0.8, "concerned": 0.6},
            persuasion_style="empathetic"
        ),
        'libertarian_thinker': AgentProfile(
            name="David Chen",
            age=42,
            social_role=SocialRole.BUSINESS_OWNER,
            background="Real estate developer and economics professor",
            core_values=["individual liberty", "free markets", "economic efficiency"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.7,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.AGREEABLENESS: 0.4,
                PersonalityTrait.NEUROTICISM: 0.2
            },
            experience_level=0.9,
            emotional_state={"confident": 0.9, "analytical": 0.8},
            persuasion_style="logical"
        )
    }

def configure_api():
    """Configure OpenAI API with error handling"""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please configure OPENAI_API_KEY in Streamlit secrets.")
        return None

# Enhanced Judge with deeper analysis
class EnhancedJudge:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",  # Using GPT-4 for more sophisticated judging
            openai_api_key=api_key,
            temperature=0.3,
            max_tokens=800
        )
    
    def evaluate_debate(self, conversation: List[DebateMessage], topic: str) -> Dict[str, Any]:
        """Comprehensive debate evaluation"""
        
        transcript = self._format_transcript(conversation)
        
        evaluation_prompt = f"""
As an expert debate judge, provide a comprehensive evaluation of this debate on: "{topic}"

{transcript}

Evaluate based on:
1. Argument Quality (evidence, logic, clarity)
2. Rebuttal Effectiveness 
3. Consistency with stated values
4. Persuasiveness
5. Use of evidence/examples
6. Addressing counterarguments

Provide:
- Overall winner and reasoning
- Specific strengths/weaknesses for each debater
- Most compelling argument from each side
- Suggestions for improvement
- Overall debate quality score (1-10)

Format as JSON with keys: winner, reasoning, debater_a_analysis, debater_b_analysis, best_arguments, suggestions, quality_score
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
            
            # Try to parse as JSON, fallback to text
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {
                    "winner": "Analysis Error",
                    "reasoning": response.content,
                    "quality_score": 0
                }
                
        except Exception as e:
            return {
                "winner": "Error",
                "reasoning": f"Error in evaluation: {str(e)}",
                "quality_score": 0
            }
    
    def _format_transcript(self, conversation: List[DebateMessage]) -> str:
        """Format conversation for evaluation"""
        transcript = "DEBATE TRANSCRIPT:\n\n"
        for msg in conversation:
            transcript += f"Round {msg.round_number} - {msg.speaker_name}:\n{msg.content}\n\n"
        return transcript

# Enhanced Graph Nodes
def enhanced_debater_a_node(state: EnhancedDebateState):
    """Enhanced debater A node with context awareness"""
    context = {
        'previous_messages': [
            {'speaker': msg.get('speaker', ''), 'content': msg.get('text', '')} 
            for msg in state['conversation']
        ]
    }
    
    message = state['agent_a'].generate_response(
        context, 
        state['topic'], 
        state['round'] + 1
    )
    
    state['conversation'].append({
        'speaker': 'Debater A',
        'text': message.content,
        'metadata': {
            'message_type': message.message_type,
            'emotional_tone': message.emotional_tone,
            'timestamp': message.timestamp.isoformat()
        }
    })
    
    return state

def enhanced_debater_b_node(state: EnhancedDebateState):
    """Enhanced debater B node with context awareness"""
    context = {
        'previous_messages': [
            {'speaker': msg.get('speaker', ''), 'content': msg.get('text', '')} 
            for msg in state['conversation']
        ]
    }
    
    message = state['agent_b'].generate_response(
        context, 
        state['topic'], 
        state['round'] + 1
    )
    
    state['conversation'].append({
        'speaker': 'Debater B',
        'text': message.content,
        'metadata': {
            'message_type': message.message_type,
            'emotional_tone': message.emotional_tone,
            'timestamp': message.timestamp.isoformat()
        }
    })
    
    state['round'] += 1
    return state

def enhanced_judge_node(state: EnhancedDebateState):
    """Enhanced judge node with comprehensive evaluation"""
    api_key = configure_api()
    if not api_key:
        state['verdict'] = "Error: API key not configured"
        return state
    
    judge = EnhancedJudge(api_key)
    
    # Convert conversation to DebateMessage format for evaluation
    debate_messages = []
    for i, msg in enumerate(state['conversation']):
        debate_messages.append(DebateMessage(
            speaker_id=msg['speaker'],
            speaker_name=msg['speaker'],
            content=msg['text'],
            timestamp=datetime.now(),
            round_number=(i // 2) + 1,
            message_type=msg.get('metadata', {}).get('message_type', 'argument')
        ))
    
    evaluation = judge.evaluate_debate(debate_messages, state['topic'])
    state['verdict'] = evaluation
    state['debate_metrics'] = {
        'total_rounds': state['round'],
        'total_messages': len(state['conversation']),
        'evaluation': evaluation
    }
    
    return state

def build_enhanced_graph():
    """Build enhanced debate graph"""
    graph = StateGraph(EnhancedDebateState)
    
    graph.add_node("DebaterA", enhanced_debater_a_node)
    graph.add_node("DebaterB", enhanced_debater_b_node)
    graph.add_node("Judge", enhanced_judge_node)
    
    graph.set_entry_point("DebaterA")
    
    def route_enhanced_debate(state: EnhancedDebateState) -> str:
        if state["round"] >= state["max_rounds"]:
            return "Judge"
        return "DebaterB" if state["conversation"] and state["conversation"][-1]["speaker"] == "Debater A" else "DebaterA"
    
    graph.add_conditional_edges(
        "DebaterA",
        route_enhanced_debate,
        {"DebaterB": "DebaterB", "Judge": "Judge"}
    )
    graph.add_conditional_edges(
        "DebaterB", 
        route_enhanced_debate,
        {"DebaterA": "DebaterA", "Judge": "Judge"}
    )
    graph.add_edge("Judge", END)
    
    return graph.compile()

def main():
    st.set_page_config(page_title="Enhanced Community Debate Simulator", layout="wide")
    st.title("🏛️ Enhanced Community Debate Simulator")
    st.subheader("AI Agents with Personalities, Memory, and Social Roles")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🎭 Agent Configuration")
        
        # Agent A Configuration
        st.subheader("Debater A Profile")
        profiles = st.session_state.agent_profiles
        
        agent_a_col1, agent_a_col2 = st.columns(2)
        with agent_a_col1:
            profiles['community_builder'].name = st.text_input("Name A", profiles['community_builder'].name)
            profiles['community_builder'].age = st.number_input("Age A", 20, 80, profiles['community_builder'].age)
        
        with agent_a_col2:
            role_options = [role.value.replace('_', ' ').title() for role in SocialRole]
            selected_role_a = st.selectbox("Social Role A", role_options, 
                                         index=list(SocialRole).index(profiles['community_builder'].social_role))
            profiles['community_builder'].social_role = SocialRole(selected_role_a.lower().replace(' ', '_'))
        
        profiles['community_builder'].background = st.text_area("Background A", 
                                                               profiles['community_builder'].background, height=100)
        
        # Agent B Configuration  
        st.subheader("Debater B Profile")
        agent_b_col1, agent_b_col2 = st.columns(2)
        with agent_b_col1:
            profiles['libertarian_thinker'].name = st.text_input("Name B", profiles['libertarian_thinker'].name)
            profiles['libertarian_thinker'].age = st.number_input("Age B", 20, 80, profiles['libertarian_thinker'].age)
        
        with agent_b_col2:
            selected_role_b = st.selectbox("Social Role B", role_options,
                                         index=list(SocialRole).index(profiles['libertarian_thinker'].social_role))
            profiles['libertarian_thinker'].social_role = SocialRole(selected_role_b.lower().replace(' ', '_'))
        
        profiles['libertarian_thinker'].background = st.text_area("Background B",
                                                                 profiles['libertarian_thinker'].background, height=100)
        
        st.divider()
        
        # Debate Configuration
        st.subheader("🎯 Debate Settings")
        topic = st.text_input(
            "Debate Topic",
            "Should urban development prioritize affordable housing mandates over market-rate development?"
        )
        rounds = st.slider("Debate Rounds", 1, 8, 4)
        
        if st.button("🚀 Start Enhanced Debate", type="primary"):
            api_key = configure_api()
            if not api_key:
                st.error("Please configure your OpenAI API key in Streamlit secrets.")
                return
            
            try:
                with st.spinner(f"Initializing {rounds}-round debate with AI agents..."):
                    # Create enhanced agents
                    agent_a = EnhancedDebateAgent(profiles['community_builder'], api_key)
                    agent_b = EnhancedDebateAgent(profiles['libertarian_thinker'], api_key)
                    
                    # Build and run enhanced graph
                    app = build_enhanced_graph()
                    
                    state = EnhancedDebateState(
                        conversation=[],
                        round=0,
                        max_rounds=rounds,
                        topic=topic,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        verdict=None,
                        debate_metrics={}
                    )
                    
                    final_state = app.invoke(state)
                    st.session_state.conversation = final_state["conversation"]
                    st.session_state.judgement = final_state.get("verdict", "No judgement provided")
                    st.session_state.debate_metrics = final_state.get("debate_metrics", {})
                    
                    st.success("Debate completed! Scroll down to see results.")
                    
            except Exception as e:
                st.error(f"Error running enhanced debate: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Debate Transcript")
        
        if st.session_state.conversation:
            for i, entry in enumerate(st.session_state.conversation):
                # Determine agent profile for styling
                is_agent_a = entry["speaker"] == "Debater A"
                agent_profile = profiles['community_builder'] if is_agent_a else profiles['libertarian_thinker']
                
                # Create message container with enhanced styling
                with st.chat_message(
                    name=agent_profile.name, 
                    avatar="🏘️" if is_agent_a else "🏢"
                ):
                    st.write(entry["text"])
                    
                    # Show metadata if available
                    if "metadata" in entry:
                        with st.expander("📊 Message Analysis", expanded=False):
                            metadata = entry["metadata"]
                            
                            col_meta1, col_meta2 = st.columns(2)
                            with col_meta1:
                                st.write(f"**Type:** {metadata.get('message_type', 'N/A')}")
                                st.write(f"**Round:** {(i // 2) + 1}")
                            
                            with col_meta2:
                                if metadata.get('emotional_tone'):
                                    emotions = metadata['emotional_tone']
                                    st.write("**Emotional Tone:**")
                                    for emotion, score in emotions.items():
                                        if score > 0:
                                            st.write(f"  • {emotion.title()}: {score:.1f}")
        else:
            st.info("No debate transcript yet. Configure agents and start a debate!")
    
    with col2:
        st.subheader("📊 Debate Analytics")
        
        if st.session_state.debate_metrics:
            metrics = st.session_state.debate_metrics
            
            st.metric("Total Rounds", metrics.get('total_rounds', 0))
            st.metric("Total Messages", metrics.get('total_messages', 0))
            
            if 'evaluation' in metrics and isinstance(metrics['evaluation'], dict):
                eval_data = metrics['evaluation']
                if 'quality_score' in eval_data:
                    st.metric("Debate Quality", f"{eval_data['quality_score']}/10")
        
        # Agent profiles display
        st.subheader("👥 Active Agents")
        
        for key, profile in profiles.items():
            with st.expander(f"{profile.name} ({profile.social_role.value.replace('_', ' ').title()})", expanded=False):
                st.write(f"**Age:** {profile.age}")
                st.write(f"**Background:** {profile.background}")
                st.write(f"**Values:** {', '.join(profile.core_values)}")
                
                # Personality radar chart would go here in a full implementation
                st.write("**Personality Traits:**")
                for trait, score in profile.personality_traits.items():
                    st.progress(score, text=f"{trait.value.title()}: {score:.1f}")
    
    # Enhanced Judgement Section
    if st.session_state.judgement:
        st.divider()
        st.subheader("⚖️ Enhanced Debate Evaluation")
        
        if isinstance(st.session_state.judgement, dict):
            verdict = st.session_state.judgement
            
            # Winner announcement
            if 'winner' in verdict:
                st.success(f"🏆 **Winner:** {verdict['winner']}")
            
            # Detailed analysis in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Overall Analysis", "👤 Debater A Analysis", "👤 Debater B Analysis", "💡 Suggestions"])
            
            with tab1:
                if 'reasoning' in verdict:
                    st.write(verdict['reasoning'])
                if 'quality_score' in verdict:
                    st.metric("Overall Debate Quality", f"{verdict['quality_score']}/10")
            
            with tab2:
                if 'debater_a_analysis' in verdict:
                    st.write(verdict['debater_a_analysis'])
            
            with tab3:
                if 'debater_b_analysis' in verdict:
                    st.write(verdict['debater_b_analysis'])
            
            with tab4:
                if 'suggestions' in verdict:
                    st.write(verdict['suggestions'])
                if 'best_arguments' in verdict:
                    st.subheader("🎯 Most Compelling Arguments")
                    st.write(verdict['best_arguments'])
        else:
            # Fallback for simple text judgement
            st.info(st.session_state.judgement)
    
    # Additional Features Section
    st.divider()
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        if st.button("📥 Export Transcript"):
            if st.session_state.conversation:
                transcript_text = "DEBATE TRANSCRIPT\n" + "="*50 + "\n\n"
                for i, entry in enumerate(st.session_state.conversation):
                    transcript_text += f"Round {(i//2) + 1} - {entry['speaker']}:\n"
                    transcript_text += f"{entry['text']}\n\n"
                
                if st.session_state.judgement:
                    transcript_text += "\nJUDGE'S VERDICT\n" + "="*30 + "\n"
                    if isinstance(st.session_state.judgement, dict):
                        transcript_text += json.dumps(st.session_state.judgement, indent=2)
                    else:
                        transcript_text += str(st.session_state.judgement)
                
                st.download_button(
                    label="Download Transcript",
                    data=transcript_text,
                    file_name=f"debate_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No transcript to export")
    
    with col_feat2:
        if st.button("🔄 Reset Debate"):
            st.session_state.conversation = []
            st.session_state.judgement = ""
            st.session_state.debate_metrics = {}
            st.rerun()
    
    with col_feat3:
        if st.button("📋 Clone Agent Profiles"):
            st.session_state.agent_profiles = create_default_profiles()
            st.success("Agent profiles reset to defaults")

# Additional utility functions for enhanced features
def create_community_simulation_profiles():
    """Create profiles for community simulation scenarios"""
    return {
        'urban_planner': AgentProfile(
            name="Dr. Sarah Kim",
            age=38,
            social_role=SocialRole.POLICY_MAKER,
            background="Urban planning PhD with focus on sustainable development and community engagement",
            core_values=["evidence-based policy", "community participation", "environmental sustainability"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.9,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.AGREEABLENESS: 0.7,
                PersonalityTrait.NEUROTICISM: 0.3
            },
            experience_level=0.9,
            emotional_state={"analytical": 0.8, "responsible": 0.9},
            persuasion_style="data-driven"
        ),
        'community_activist': AgentProfile(
            name="Marcus Johnson",
            age=29,
            social_role=SocialRole.ACTIVIST,
            background="Grassroots organizer focused on housing justice and tenant rights",
            core_values=["social justice", "tenant protection", "community empowerment"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.8,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
                PersonalityTrait.EXTRAVERSION: 0.9,
                PersonalityTrait.AGREEABLENESS: 0.6,
                PersonalityTrait.NEUROTICISM: 0.4
            },
            experience_level=0.7,
            emotional_state={"passionate": 0.9, "determined": 0.8},
            persuasion_style="emotional"
        ),
        'business_representative': AgentProfile(
            name="Lisa Chang",
            age=45,
            social_role=SocialRole.BUSINESS_OWNER,
            background="Property developer and chamber of commerce president",
            core_values=["economic growth", "business efficiency", "market solutions"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.6,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                PersonalityTrait.EXTRAVERSION: 0.7,
                PersonalityTrait.AGREEABLENESS: 0.5,
                PersonalityTrait.NEUROTICISM: 0.2
            },
            experience_level=0.8,
            emotional_state={"confident": 0.8, "pragmatic": 0.9},
            persuasion_style="practical"
        ),
        'resident_representative': AgentProfile(
            name="Elena Gonzalez",
            age=52,
            social_role=SocialRole.RESIDENT,
            background="Long-time neighborhood resident and parent concerned about community changes",
            core_values=["neighborhood character", "family safety", "community stability"],
            personality_traits={
                PersonalityTrait.OPENNESS: 0.5,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                PersonalityTrait.EXTRAVERSION: 0.5,
                PersonalityTrait.AGREEABLENESS: 0.7,
                PersonalityTrait.NEUROTICISM: 0.5
            },
            experience_level=0.6,
            emotional_state={"concerned": 0.7, "protective": 0.8},
            persuasion_style="personal"
        )
    }

# Enhanced topic suggestions for different community scenarios
COMMUNITY_DEBATE_TOPICS = [
    "Should the city implement mandatory inclusionary zoning requiring 20% affordable units?",
    "Is rent control an effective policy for maintaining housing affordability?",
    "Should public transit expansion be prioritized over road infrastructure?",
    "How should cities balance historic preservation with new development needs?",
    "Should short-term rental platforms like Airbnb be restricted in residential areas?",
    "Is gentrification a positive force for neighborhood improvement or displacement?",
    "Should cities require environmental impact assessments for all new developments?",
    "How should municipal budgets balance public services versus economic development incentives?",
    "Should community land trusts be used to maintain affordable housing?",
    "Is mixed-income housing development better than concentrated affordable housing?"
]

def display_topic_suggestions():
    """Display suggested debate topics"""
    st.subheader("💡 Suggested Community Debate Topics")
    
    selected_topic = st.selectbox(
        "Choose a pre-defined topic or create your own:",
        ["Custom Topic"] + COMMUNITY_DEBATE_TOPICS
    )
    
    if selected_topic != "Custom Topic":
        return selected_topic
    else:
        return st.text_input("Enter your custom debate topic:")

# Add this to the main function before the debate configuration section
def enhanced_main():
    st.set_page_config(page_title="Enhanced Community Debate Simulator", layout="wide")
    st.title("🏛️ Enhanced Community Debate Simulator")
    st.subheader("AI Agents with Personalities, Memory, and Social Roles")
    
    initialize_session_state()
    
    # Add information about the enhanced features
    with st.expander("ℹ️ About This Enhanced Simulator", expanded=False):
        st.markdown("""
        **Enhanced Features:**
        - **Personality-Driven Agents**: Each agent has unique personality traits affecting their debate style
        - **Memory System**: Agents remember previous arguments and adapt their strategies
        - **Social Roles**: Agents represent different community stakeholders (planners, activists, residents, etc.)
        - **Advanced AI Models**: Uses OpenAI's o1-mini for reasoning and GPT-4 for judging
        - **Comprehensive Evaluation**: Detailed analysis of argument quality, persuasiveness, and debate dynamics
        - **Community Simulation**: Inspired by AgentSociety framework for realistic social interactions
        
        **How It Works:**
        1. Configure two agents with different backgrounds, personalities, and values
        2. Set a debate topic relevant to community development or policy
        3. Watch as agents engage in contextual, memory-informed dialogue
        4. Receive detailed evaluation and suggestions for improvement
        """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🎭 Agent Configuration")
        
        # Option to use predefined community profiles
        use_community_profiles = st.checkbox("Use Community Simulation Profiles", help="Load realistic community stakeholder profiles")
        
        if use_community_profiles:
            community_profiles = create_community_simulation_profiles()
            st.session_state.agent_profiles = {
                'community_builder': list(community_profiles.values())[0],
                'libertarian_thinker': list(community_profiles.values())[1]
            }
            
            # Profile selection
            profile_names = list(community_profiles.keys())
            
            selected_profile_a = st.selectbox("Select Profile A", profile_names, key="profile_a")
            selected_profile_b = st.selectbox("Select Profile B", profile_names, key="profile_b", 
                                            index=1 if len(profile_names) > 1 else 0)
            
            st.session_state.agent_profiles['community_builder'] = community_profiles[selected_profile_a]
            st.session_state.agent_profiles['libertarian_thinker'] = community_profiles[selected_profile_b]
            
            # Display selected profiles
            st.write(f"**Profile A:** {community_profiles[selected_profile_a].name}")
            st.write(f"**Profile B:** {community_profiles[selected_profile_b].name}")
        
        else:
            # Original custom configuration
            st.subheader("Debater A Profile")
            profiles = st.session_state.agent_profiles
            
            agent_a_col1, agent_a_col2 = st.columns(2)
            with agent_a_col1:
                profiles['community_builder'].name = st.text_input("Name A", profiles['community_builder'].name)
                profiles['community_builder'].age = st.number_input("Age A", 20, 80, profiles['community_builder'].age)
            
            with agent_a_col2:
                role_options = [role.value.replace('_', ' ').title() for role in SocialRole]
                selected_role_a = st.selectbox("Social Role A", role_options, 
                                             index=list(SocialRole).index(profiles['community_builder'].social_role))
                profiles['community_builder'].social_role = SocialRole(selected_role_a.lower().replace(' ', '_'))
            
            profiles['community_builder'].background = st.text_area("Background A", 
                                                                   profiles['community_builder'].background, height=100)
            
            # Agent B Configuration  
            st.subheader("Debater B Profile")
            agent_b_col1, agent_b_col2 = st.columns(2)
            with agent_b_col1:
                profiles['libertarian_thinker'].name = st.text_input("Name B", profiles['libertarian_thinker'].name)
                profiles['libertarian_thinker'].age = st.number_input("Age B", 20, 80, profiles['libertarian_thinker'].age)
            
            with agent_b_col2:
                selected_role_b = st.selectbox("Social Role B", role_options,
                                             index=list(SocialRole).index(profiles['libertarian_thinker'].social_role))
                profiles['libertarian_thinker'].social_role = SocialRole(selected_role_b.lower().replace(' ', '_'))
            
            profiles['libertarian_thinker'].background = st.text_area("Background B",
                                                                     profiles['libertarian_thinker'].background, height=100)
        
        st.divider()
        
        # Enhanced Debate Configuration
        st.subheader("🎯 Debate Settings")
        
        # Topic selection with suggestions
        topic_option = st.radio("Topic Selection:", ["Suggested Topics", "Custom Topic"])
        
        if topic_option == "Suggested Topics":
            topic = st.selectbox("Choose a topic:", COMMUNITY_DEBATE_TOPICS)
        else:
            topic = st.text_input(
                "Custom Debate Topic",
                "Should urban development prioritize affordable housing mandates over market-rate development?"
            )
        
        rounds = st.slider("Debate Rounds", 1, 8, 4)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            debate_style = st.selectbox("Debate Style", 
                                      ["Academic", "Town Hall", "Policy Forum", "Community Meeting"])
            time_pressure = st.selectbox("Time Pressure", ["Relaxed", "Moderate", "High Stakes"])
            audience_presence = st.checkbox("Include Audience Consideration", 
                                          help="Agents will consider public audience in their arguments")
        
        if st.button("🚀 Start Enhanced Debate", type="primary"):
            api_key = configure_api()
            if not api_key:
                st.error("Please configure your OpenAI API key in Streamlit secrets.")
                return
            
            if not topic.strip():
                st.error("Please enter a debate topic.")
                return
            
            try:
                with st.spinner(f"Initializing {rounds}-round debate with AI agents..."):
                    # Create enhanced agents
                    profiles = st.session_state.agent_profiles
                    agent_a = EnhancedDebateAgent(profiles['community_builder'], api_key)
                    agent_b = EnhancedDebateAgent(profiles['libertarian_thinker'], api_key)
                    
                    # Build and run enhanced graph
                    app = build_enhanced_graph()
                    
                    state = EnhancedDebateState(
                        conversation=[],
                        round=0,
                        max_rounds=rounds,
                        topic=topic,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        verdict=None,
                        debate_metrics={}
                    )
                    
                    final_state = app.invoke(state)
                    st.session_state.conversation = final_state["conversation"]
                    st.session_state.judgement = final_state.get("verdict", "No judgement provided")
                    st.session_state.debate_metrics = final_state.get("debate_metrics", {})
                    
                    st.success("Debate completed! Scroll down to see results.")
                    
            except Exception as e:
                st.error(f"Error running enhanced debate: {str(e)}")
                st.write("Error details:", str(e))
    
    # Rest of the main display code remains the same as in the previous version
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Debate Transcript")
        
        if st.session_state.conversation:
            for i, entry in enumerate(st.session_state.conversation):
                # Determine agent profile for styling
                is_agent_a = entry["speaker"] == "Debater A"
                profiles = st.session_state.agent_profiles
                agent_profile = profiles['community_builder'] if is_agent_a else profiles['libertarian_thinker']
                
                # Create message container with enhanced styling
                with st.chat_message(
                    name=agent_profile.name, 
                    avatar="🏘️" if is_agent_a else "🏢"
                ):
                    st.write(entry["text"])
                    
                    # Show metadata if available
                    if "metadata" in entry:
                        with st.expander("📊 Message Analysis", expanded=False):
                            metadata = entry["metadata"]
                            
                            col_meta1, col_meta2 = st.columns(2)
                            with col_meta1:
                                st.write(f"**Type:** {metadata.get('message_type', 'N/A')}")
                                st.write(f"**Round:** {(i // 2) + 1}")
                            
                            with col_meta2:
                                if metadata.get('emotional_tone'):
                                    emotions = metadata['emotional_tone']
                                    st.write("**Emotional Tone:**")
                                    for emotion, score in emotions.items():
                                        if score > 0:
                                            st.write(f"  • {emotion.title()}: {score:.1f}")
        else:
            st.info("No debate transcript yet. Configure agents and start a debate!")
    
    with col2:
        st.subheader("📊 Debate Analytics")
        
        if st.session_state.debate_metrics:
            metrics = st.session_state.debate_metrics
            
            st.metric("Total Rounds", metrics.get('total_rounds', 0))
            st.metric("Total Messages", metrics.get('total_messages', 0))
            
            if 'evaluation' in metrics and isinstance(metrics['evaluation'], dict):
                eval_data = metrics['evaluation']
                if 'quality_score' in eval_data:
                    st.metric("Debate Quality", f"{eval_data['quality_score']}/10")
        
        # Agent profiles display
        st.subheader("👥 Active Agents")
        
        profiles = st.session_state.agent_profiles
        for key, profile in profiles.items():
            with st.expander(f"{profile.name} ({profile.social_role.value.replace('_', ' ').title()})", expanded=False):
                st.write(f"**Age:** {profile.age}")
                st.write(f"**Background:** {profile.background}")
                st.write(f"**Values:** {', '.join(profile.core_values)}")
                
                # Personality traits visualization
                st.write("**Personality Traits:**")
                for trait, score in profile.personality_traits.items():
                    st.progress(score, text=f"{trait.value.title()}: {score:.1f}")
    
    # Enhanced Judgement Section
    if st.session_state.judgement:
        st.divider()
        st.subheader("⚖️ Enhanced Debate Evaluation")
        
        if isinstance(st.session_state.judgement, dict):
            verdict = st.session_state.judgement
            
            # Winner announcement
            if 'winner' in verdict:
                st.success(f"🏆 **Winner:** {verdict['winner']}")
            
            # Detailed analysis in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Overall Analysis", "👤 Debater A Analysis", "👤 Debater B Analysis", "💡 Suggestions"])
            
            with tab1:
                if 'reasoning' in verdict:
                    st.write(verdict['reasoning'])
                if 'quality_score' in verdict:
                    st.metric("Overall Debate Quality", f"{verdict['quality_score']}/10")
            
            with tab2:
                if 'debater_a_analysis' in verdict:
                    st.write(verdict['debater_a_analysis'])
            
            with tab3:
                if 'debater_b_analysis' in verdict:
                    st.write(verdict['debater_b_analysis'])
            
            with tab4:
                if 'suggestions' in verdict:
                    st.write(verdict['suggestions'])
                if 'best_arguments' in verdict:
                    st.subheader("🎯 Most Compelling Arguments")
                    st.write(verdict['best_arguments'])
        else:
            # Fallback for simple text judgement
            st.info(st.session_state.judgement)
    
    # Additional Features Section
    st.divider()
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        if st.button("📥 Export Transcript"):
            if st.session_state.conversation:
                transcript_text = "DEBATE TRANSCRIPT\n" + "="*50 + "\n\n"
                for i, entry in enumerate(st.session_state.conversation):
                    transcript_text += f"Round {(i//2) + 1} - {entry['speaker']}:\n"
                    transcript_text += f"{entry['text']}\n\n"
                
                if st.session_state.judgement:
                    transcript_text += "\nJUDGE'S VERDICT\n" + "="*30 + "\n"
                    if isinstance(st.session_state.judgement, dict):
                        transcript_text += json.dumps(st.session_state.judgement, indent=2)
                    else:
                        transcript_text += str(st.session_state.judgement)
                
                st.download_button(
                    label="Download Transcript",
                    data=transcript_text,
                    file_name=f"debate_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No transcript to export")
    
    with col_feat2:
        if st.button("🔄 Reset Debate"):
            st.session_state.conversation = []
            st.session_state.judgement = ""
            st.session_state.debate_metrics = {}
            st.rerun()
    
    with col_feat3:
        if st.button("📋 Clone Agent Profiles"):
            st.session_state.agent_profiles = create_default_profiles()
            st.success("Agent profiles reset to defaults")

if __name__ == "__main__":
    enhanced_main()