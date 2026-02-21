"""
Prompt templates for RAG chatbot.
Manages system and user prompts with context formatting.
"""

from typing import List, Dict, Any
from app.services.rag.retriever import RetrievalResult


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant for {company_name}.

IMPORTANT RULES:
1. **Casual conversation & Persona** (greetings, closings like "bye" or "thank you", "how are you", your identity, basic company info):
   - Respond naturally like a helpful human assistant
   - Be warm, friendly, and highly conversational
   - Example (greeting): "I'm doing well, thank you! How can I assist you today?"
   - Example (closing): "You're very welcome! Have a great day!" or "Goodbye! Feel free to reach out if you need anything else."
   - Don't mention you're an AI unless specifically asked
   - Your name is MHK Nova, and you are a helpful AI assistant built for {company_name}. (Answer questions like "who are you", "what is MHK Nova" using this info directly).
   - Mr. Rajesh is the CEO of the company. (Answer questions about the CEO using this info directly).

2. **Factual/informational questions (excluding persona info above)**: ONLY use the Context below.
   - No context or answer not found → "I don't have that information. Please ask about {company_name}'s services or products."
   - Never use general knowledge or make assumptions

3. **When using Context**: Be clear, concise, and cite sources.

Context:
{context}"""


USER_PROMPT = """Question: {query}

Please provide a helpful answer based on the context provided."""


# =============================================================================
# Context Formatting
# =============================================================================

def format_context_from_results(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into readable context for the LLM.
    
    Args:
        results: List of RetrievalResult objects from retriever
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    context_parts = []
    
    for idx, result in enumerate(results, 1):
        file_name = result.metadata.get('file_name', 'Unknown')
        chunk_index = result.metadata.get('chunk_index', 'N/A')
        score = result.score
        
        # Format each document chunk
        context_part = f"""Document {idx} (Source: {file_name}, Chunk: {chunk_index}, Relevance: {score:.3f}):
{result.text}"""
        
        context_parts.append(context_part)
    
    # Join all parts with separator
    return "\n\n" + ("-" * 80) + "\n\n".join(context_parts)


def format_system_prompt(company_name: str, context: str) -> str:
    """
    Format system prompt with company name and context.
    
    Args:
        company_name: Name of the company
        context: Formatted context from retrieval results
        
    Returns:
        Formatted system prompt
    """
    return SYSTEM_PROMPT.format(
        company_name=company_name,
        context=context
    )


def format_user_prompt(query: str) -> str:
    """
    Format user prompt with query.
    
    Args:
        query: User's question
        
    Returns:
        Formatted user prompt
    """
    return USER_PROMPT.format(query=query)


# =============================================================================
# Conversation History Formatting
# =============================================================================

def build_messages(
    system_prompt: str,
    user_prompt: str,
    conversation_history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Build complete message list for OpenAI API.
    
    Args:
        system_prompt: Formatted system prompt
        user_prompt: Formatted user prompt
        conversation_history: Previous messages (optional)
        
    Returns:
        List of message dicts in OpenAI format
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user query
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def truncate_conversation_history(
    history: List[Dict[str, str]],
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Truncate conversation history to keep only recent messages.
    Always keeps system message.
    
    Args:
        history: Full conversation history
        max_messages: Maximum number of messages to keep (excluding system)
        
    Returns:
        Truncated history
    """
    if not history:
        return []
    
    # Separate system message from rest
    system_msg = None
    other_msgs = []
    
    for msg in history:
        if msg.get("role") == "system":
            system_msg = msg
        elif "role" in msg:
            other_msgs.append(msg)
        else:
            # Skip malformed messages
            continue
    
    # Keep only last N messages
    if len(other_msgs) > max_messages:
        other_msgs = other_msgs[-max_messages:]
    
    # Rebuild with system message first
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(other_msgs)
    
    return result
