"""
Pesticide Database - Structured data loading from markdown table
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PesticideDatabase:
    """
    Parse and query pesticide recommendations from markdown table
    """
    
    def __init__(self, md_path: str = "knowledge_base/pesticide_recommendations.md"):
        self.md_path = Path(md_path)
        self.data: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}
        self.all_crops: List[str] = []
        self.all_pests: List[str] = []
        self.all_applications: List[str] = []
        
        self._load_and_parse()
    
    def _load_and_parse(self):
        """Load markdown file and parse table into structured dict"""
        if not self.md_path.exists():
            logger.error(f"Markdown file not found: {self.md_path}")
            return
        
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract table rows (skip header and separator)
        lines = content.strip().split('\n')
        table_lines = [l for l in lines if l.strip().startswith('|')]
        
        if len(table_lines) < 3:
            logger.error("Invalid table format")
            return
        
        # Skip header (line 0) and separator (line 1)
        data_lines = table_lines[2:]
        
        for line in data_lines:
            try:
                row = self._parse_table_row(line)
                if row:
                    self._add_to_structure(row)
            except Exception as e:
                logger.warning(f"Error parsing row: {line[:50]}... | Error: {e}")
        
        # Extract unique values
        self.all_crops = sorted(set(self.data.keys()))
        self.all_pests = sorted(set(self._get_all_pests()))
        self.all_applications = sorted(set(self._get_all_applications()))
        
        logger.info(f"✅ Loaded {len(self.all_crops)} crops, {len(self.all_pests)} pests, {len(self.all_applications)} application types")
    
    def _parse_table_row(self, line: str) -> Optional[Dict]:
        """Parse a single table row into a dict"""
        parts = [p.strip() for p in line.split('|')]
        
        # Remove empty first/last elements from split
        parts = [p for p in parts if p]
        
        if len(parts) < 7:
            return None
        
        return {
            'crop': parts[0],
            'problem_type': parts[1],
            'pest_name': parts[2],
            'solution': parts[3],
            'application': parts[4],
            'dosage': parts[5],
            'waiting_period': parts[6]
        }
    
    def _add_to_structure(self, row: Dict):
        """Add parsed row to nested data structure"""
        crop = row['crop']
        problem_type = row['problem_type']
        pest_name = row['pest_name']
        
        # Initialize nested structure
        if crop not in self.data:
            self.data[crop] = {}
        
        if problem_type not in self.data[crop]:
            self.data[crop][problem_type] = {}
        
        if pest_name not in self.data[crop][problem_type]:
            self.data[crop][problem_type][pest_name] = []
        
        # Add solution
        self.data[crop][problem_type][pest_name].append({
            'solution': row['solution'],
            'application': row['application'],
            'dosage': row['dosage'],
            'waiting_period': row['waiting_period']
        })
    
    def _get_all_pests(self) -> List[str]:
        """Extract all unique pest names"""
        pests = []
        for crop_data in self.data.values():
            for problem_data in crop_data.values():
                pests.extend(problem_data.keys())
        return pests
    
    def _get_all_applications(self) -> List[str]:
        """Extract all unique application types"""
        applications = []
        for crop_data in self.data.values():
            for problem_data in crop_data.values():
                for solutions in problem_data.values():
                    applications.extend([s['application'] for s in solutions])
        return applications
    
    def _fuzzy_match(self, query: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """Fuzzy match query against candidates using sequence matching"""
        if not query or not candidates:
            return None
        
        query_lower = query.lower().strip()
        best_match = None
        best_ratio = 0.0
        
        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            
            # Exact match
            if query_lower == candidate_lower:
                return candidate
            
            # Substring match
            if query_lower in candidate_lower or candidate_lower in query_lower:
                ratio = 0.9
            else:
                # Sequence similarity
                ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    # Public API Methods
    
    def get_all_crops(self) -> List[str]:
        """Get list of all crops"""
        return self.all_crops
    
    def get_problem_types(self, crop: str) -> List[str]:
        """Get problem types for a crop"""
        if crop not in self.data:
            return []
        return list(self.data[crop].keys())
    
    def get_pests(self, crop: str, problem_type: Optional[str] = None) -> List[str]:
        """Get all pests for a crop (optionally filtered by problem type)"""
        if crop not in self.data:
            return []
        
        if problem_type and problem_type in self.data[crop]:
            return list(self.data[crop][problem_type].keys())
        
        # Get all pests for this crop across all problem types
        pests = []
        for problem_data in self.data[crop].values():
            pests.extend(problem_data.keys())
        
        return sorted(set(pests))
    
    def get_application_types(self, crop: str, pest: str) -> List[str]:
        """Get available application types for crop + pest combination"""
        if crop not in self.data:
            return []
        
        applications = []
        for problem_data in self.data[crop].values():
            if pest in problem_data:
                applications.extend([s['application'] for s in problem_data[pest]])
        
        return sorted(set(applications))
    
    def get_solutions(
        self, 
        crop: str, 
        pest: Optional[str] = None, 
        application_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get solutions filtered by crop, pest, and optionally application type
        Returns list of solution dicts with metadata
        """
        if crop not in self.data:
            return []
        
        solutions = []
        
        for problem_type, problem_data in self.data[crop].items():
            if pest:
                # Specific pest
                if pest in problem_data:
                    for sol in problem_data[pest]:
                        if application_type and sol['application'] != application_type:
                            continue
                        
                        solutions.append({
                            'crop': crop,
                            'problem_type': problem_type,
                            'pest_name': pest,
                            **sol
                        })
            else:
                # All pests for this crop
                for pest_name, pest_solutions in problem_data.items():
                    for sol in pest_solutions:
                        if application_type and sol['application'] != application_type:
                            continue
                        
                        solutions.append({
                            'crop': crop,
                            'problem_type': problem_type,
                            'pest_name': pest_name,
                            **sol
                        })
        
        return solutions
    
    def fuzzy_match_crop(self, query: str) -> Optional[str]:
        """Fuzzy match crop name"""
        return self._fuzzy_match(query, self.all_crops)
    
    def fuzzy_match_pest(self, query: str, crop: Optional[str] = None) -> Optional[str]:
        """Fuzzy match pest name (optionally within a crop)"""
        if crop:
            pests = self.get_pests(crop)
        else:
            pests = self.all_pests
        
        return self._fuzzy_match(query, pests)
    
    def fuzzy_match_application_type(self, query: str) -> Optional[str]:
        """Fuzzy match application type"""
        return self._fuzzy_match(query, self.all_applications)
    
    def get_most_common_crops(self, limit: int = 10) -> List[str]:
        """Get most common crops (by number of solutions)"""
        crop_counts = [(crop, len(self.get_solutions(crop))) for crop in self.all_crops]
        crop_counts.sort(key=lambda x: x[1], reverse=True)
        return [crop for crop, _ in crop_counts[:limit]]