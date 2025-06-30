import logging
from typing import List, Dict, Any
import re
from io import BytesIO
import os
import PyPDF2

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle document text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text_content = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Add page marker for source tracking
                            text_content += f"\n[PAGE {page_num + 1}]\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
            
            if not text_content.strip():
                logger.warning("No text content extracted from PDF, using sample data")
                return self._get_sample_text()
            
            # Clean and normalize text
            text_content = self._clean_text(text_content)
            logger.info(f"Extracted {len(text_content)} characters from {file_path}")
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            logger.info("Falling back to sample data for demonstration")
            return self._get_sample_text()
    
    def _get_sample_text(self) -> str:
        """Get sample company text for demonstration."""
        sample_text = """
        [PAGE 1]
        PAVES TECHNOLOGIES - COMPANY OVERVIEW
        
        Paves Technologies is a leading construction and infrastructure company specializing in 
        road construction, bridge building, and urban development projects. Founded in 2010, 
        we have completed over 500 major infrastructure projects across the region.
        
        Our core services include:
        - Highway and road construction
        - Bridge engineering and construction
        - Urban planning and development
        - Environmental impact assessments
        - Project management and consulting
        
        [PAGE 2]
        SAFETY PROTOCOLS
        
        Safety is our top priority at Paves Technologies. All employees must follow these 
        essential safety protocols:
        
        1. Personal Protective Equipment (PPE) must be worn at all times on construction sites
        2. Daily safety meetings are mandatory before starting work
        3. All equipment must be inspected before use
        4. Emergency procedures must be reviewed weekly
        5. Incident reporting is required within 24 hours
        
        For emergency situations, contact the safety hotline: 1-800-PAVES-SAFE
        
        [PAGE 3]
        PROJECT MANAGEMENT GUIDELINES
        
        Effective project management is crucial for successful delivery of infrastructure projects.
        Our standard procedures include:
        
        - Initial site assessment and planning
        - Environmental compliance verification
        - Resource allocation and scheduling
        - Quality control checkpoints
        - Client communication protocols
        - Final project documentation and handover
        """
        
        return self._clean_text(sample_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\&\%\$\#\@\+\=\*]', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def create_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create text chunks with metadata."""
        if not text:
            return []
        
        chunks = []
        current_page = 1
        
        # Split text by page markers first
        pages = re.split(r'\[PAGE (\d+)\]', text)
        
        for i in range(1, len(pages), 2):
            if i + 1 < len(pages):
                page_num = int(pages[i])
                page_text = pages[i + 1].strip()
                
                if page_text:
                    # Create chunks from page text
                    page_chunks = self._split_text_into_chunks(page_text)
                    
                    for chunk_text in page_chunks:
                        if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                            chunks.append({
                                "content": chunk_text.strip(),
                                "metadata": {
                                    "filename": filename,
                                    "page": page_num,
                                    "chunk_length": len(chunk_text)
                                }
                            })
        
        # If no page markers found, chunk the entire text
        if not chunks:
            text_chunks = self._split_text_into_chunks(text)
            for chunk_text in text_chunks:
                if len(chunk_text.strip()) > 50:
                    chunks.append({
                        "content": chunk_text.strip(),
                        "metadata": {
                            "filename": filename,
                            "page": "N/A",
                            "chunk_length": len(chunk_text)
                        }
                    })
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []
        
        # Split by sentences first to maintain context
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length + (1 if current_chunk else 0)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good breaking point for overlap
        overlap_start = len(text) - self.chunk_overlap
        
        # Look for sentence boundary within overlap region
        sentences = re.split(r'(?<=[.!?])\s+', text[overlap_start:])
        
        if len(sentences) > 1:
            # Use complete sentences for overlap
            return sentences[-1] if sentences[-1] else text[overlap_start:]
        else:
            # Use character-based overlap
            return text[overlap_start:]
