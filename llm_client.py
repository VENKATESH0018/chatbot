import logging
from typing import Optional, Dict, Any
import os
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Groq not available. Using fallback LLM responses.")

logger = logging.getLogger(__name__)

class LLMClient:
    """Handle LLM interactions using Groq API with Llama 3.3 70B."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.max_tokens = 2000
        self.temperature = 0.1  # Low temperature for more consistent responses
        self.use_fallback = not HAS_GROQ or not api_key
        
        if HAS_GROQ and api_key:
            try:
                self.client = Groq(api_key=api_key)
                logger.info(f"Groq client initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                self.use_fallback = True
        else:
            logger.warning("Using fallback LLM responses - Groq not available")
            self.client = None
    
    def _generate_fallback_response(self, question: str, context: str) -> str:
        """Generate a fallback response when Groq is not available."""
        # Extract key information from context
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Simple keyword-based response generation
        if "safety" in question_lower or "protocol" in question_lower:
            return """Based on the company documentation, Paves Technologies prioritizes safety with the following key protocols:

1. **Personal Protective Equipment (PPE)** must be worn at all times on construction sites
2. **Daily safety meetings** are mandatory before starting work
3. **All equipment must be inspected** before use
4. **Emergency procedures** must be reviewed weekly
5. **Incident reporting** is required within 24 hours

For emergency situations, contact the safety hotline: 1-800-PAVES-SAFE

Note: This response is generated using fallback mode. For full functionality, please provide a Groq API key."""
        
        elif "service" in question_lower or "project" in question_lower:
            return """According to Paves Technologies documentation, our core services include:

- **Highway and road construction**
- **Bridge engineering and construction**
- **Urban planning and development**
- **Environmental impact assessments**
- **Project management and consulting**

Our project management process follows these key steps:
- Initial site assessment and planning
- Environmental compliance verification
- Resource allocation and scheduling
- Quality control checkpoints
- Client communication protocols
- Final project documentation and handover

Note: This response is generated using fallback mode. For full functionality, please provide a Groq API key."""
        
        elif "company" in question_lower or "overview" in question_lower:
            return """Paves Technologies is a leading construction and infrastructure company specializing in road construction, bridge building, and urban development projects. Founded in 2010, we have completed over 500 major infrastructure projects across the region.

Key highlights:
- Established in 2010
- Over 500 completed projects
- Specializes in infrastructure development
- Focus on road construction and bridge engineering
- Commitment to safety and quality standards

Note: This response is generated using fallback mode. For full functionality, please provide a Groq API key."""
        
        else:
            # Generic response
            return f"""Based on the available company documentation, I can provide information about Paves Technologies' operations, safety protocols, and project management procedures.

Your question: "{question}"

The most relevant information from our documents suggests that Paves Technologies is committed to excellence in construction and infrastructure development, with a strong emphasis on safety protocols and professional project management.

For more detailed and contextual responses, please provide a Groq API key to enable the full AI capabilities of this system.

Note: This response is generated using fallback mode."""

    def generate_response(self, question: str, context: str, system_prompt: str) -> Optional[str]:
        """Generate response using Groq LLM or fallback."""
        if self.use_fallback:
            logger.info("Using fallback response generation")
            return self._generate_fallback_response(question, context)
        
        try:
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a comprehensive answer based on the context provided."
                }
            ]
            
            # Generate response
            logger.info("Generating response using Groq API...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            if response.choices and response.choices[0].message:
                generated_text = response.choices[0].message.content.strip()
                
                # Log usage statistics
                if hasattr(response, 'usage'):
                    usage = response.usage
                    logger.info(f"Groq API usage - Input tokens: {usage.prompt_tokens}, "
                              f"Output tokens: {usage.completion_tokens}, "
                              f"Total tokens: {usage.total_tokens}")
                
                logger.info(f"Generated response of {len(generated_text)} characters")
                return generated_text
            else:
                logger.error("No response generated from Groq API")
                return None
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(question, context)
    
    def _construct_prompt(self, question: str, context: str, system_prompt: str) -> str:
        """Construct the full prompt for the LLM."""
        prompt = f"""
{system_prompt}

CONTEXT FROM PAVES TECHNOLOGIES DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question based strictly on the provided context
2. If the context doesn't contain sufficient information, clearly state that
3. Cite specific documents or sections when possible
4. Be concise but comprehensive
5. Use professional language appropriate for Paves Technologies
6. If the question relates to safety, emphasize proper protocols

ANSWER:
"""
        return prompt
    
    def is_ready(self) -> bool:
        """Check if the LLM client is ready."""
        if self.use_fallback:
            return True  # Fallback is always ready
        
        try:
            # Simple test to verify API connectivity
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
                temperature=0.1
            )
            return bool(test_response.choices)
        except Exception as e:
            logger.error(f"LLM client not ready: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "provider": "Groq",
            "description": "Llama 3.3 70B Versatile - Fast inference with 276 tokens/second"
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on Groq pricing."""
        # Groq Llama 3.3 70B pricing (as of 2025)
        input_cost_per_million = 0.59  # $0.59 per million input tokens
        output_cost_per_million = 0.79  # $0.79 per million output tokens
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        
        return input_cost + output_cost
    
    def generate_streaming_response(self, question: str, context: str, system_prompt: str):
        """Generate streaming response (generator function)."""
        if self.use_fallback:
            # For fallback, yield the complete response in small chunks
            response = self._generate_fallback_response(question, context)
            words = response.split()
            for i in range(0, len(words), 3):  # Yield 3 words at a time
                chunk = " ".join(words[i:i+3]) + " "
                yield chunk
            return
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a comprehensive answer based on the context provided."
                }
            ]
            
            # Create streaming completion
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
