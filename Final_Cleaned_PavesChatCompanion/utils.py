import logging
import re
from typing import List, Dict, Any
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    log_filename = os.path.join(log_dir, f"paves_rag_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("PavesRAG")
    logger.info("Logging initialized")
    
    return logger

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\&\%\$\#\@\+\=\*\n]', ' ', text)
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    return text.strip()

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source citations for display."""
    if not sources:
        return "No sources available."
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        filename = source.get("filename", "Unknown")
        page = source.get("page", "N/A")
        score = source.get("score", 0.0)
        
        formatted_sources.append(f"{i}. {filename} (Page {page}) - Relevance: {score:.2f}")
    
    return "\n".join(formatted_sources)

def validate_pdf_file(file_path: str) -> bool:
    """Validate if file is a valid PDF."""
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return False
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(file_path)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False
        
        # Try to read first few bytes to check PDF signature
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                return False
        
        return True
        
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "untitled"
    
    # Limit filename length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext
    
    return filename

def calculate_chunk_overlap(text_length: int, chunk_size: int, overlap_percentage: float = 0.1) -> int:
    """Calculate optimal chunk overlap based on text length."""
    base_overlap = int(chunk_size * overlap_percentage)
    
    # Adjust overlap based on text length
    if text_length < 1000:
        return min(base_overlap, 50)  # Small text, small overlap
    elif text_length > 10000:
        return min(base_overlap, 200)  # Large text, reasonable overlap
    else:
        return base_overlap

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key terms from text for metadata."""
    if not text:
        return []
    
    # Simple keyword extraction - could be enhanced with NLP libraries
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    
    # Filter common words
    stop_words = {
        'the', 'and', 'are', 'for', 'with', 'this', 'that', 'was', 'were',
        'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'from', 'into', 'onto',
        'upon', 'over', 'under', 'above', 'below', 'between', 'among'
    }
    
    # Count word frequency
    word_count = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]

def format_response_with_citations(response: str, sources: List[Dict[str, Any]]) -> str:
    """Format response with inline citations."""
    if not sources:
        return response
    
    # Add source references at the end
    citation_text = "\n\n**Sources:**\n"
    for i, source in enumerate(sources, 1):
        filename = source.get("filename", "Unknown")
        page = source.get("page", "N/A")
        citation_text += f"[{i}] {filename}, Page {page}\n"
    
    return response + citation_text

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from file."""
    try:
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "filename": os.path.basename(file_path)
        }
    except Exception:
        return {}

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes."""
    if not text:
        return 0
    
    word_count = len(text.split())
    reading_time = max(1, word_count // words_per_minute)
    
    return reading_time

def validate_query(query: str) -> bool:
    """Validate user query."""
    if not query or not query.strip():
        return False
    
    # Check minimum length
    if len(query.strip()) < 3:
        return False
    
    # Check maximum length
    if len(query) > 1000:
        return False
    
    # Check for potentially malicious content
    suspicious_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'document\.',
        r'window\.',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    
    return True
