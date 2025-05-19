import json
from pathlib import Path
import argparse
from typing import List, Dict, Any
import textwrap

class PhilosophyQuery:
    def __init__(self, processed_data_path: str):
        self.data_path = Path(processed_data_path)
        self.texts = self._load_processed_texts()
        
    def _load_processed_texts(self) -> List[Dict[str, Any]]:
        """Load the processed texts from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processed texts: {str(e)}")
            return []
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for texts containing the keyword in any section."""
        results = []
        keyword = keyword.lower()
        
        for text in self.texts:
            # Search in all sections
            for section, content in text.items():
                if section in ['source', 'format']:
                    continue
                    
                if isinstance(content, str) and keyword in content.lower():
                    results.append(text)
                    break
                elif isinstance(content, list):
                    for item in content:
                        if keyword in item.lower():
                            results.append(text)
                            break
                            
        return results
    
    def get_summary(self, text: Dict[str, Any]) -> str:
        """Generate a summary of a philosophical text."""
        summary = []
        
        # Add source information
        summary.append(f"Source: {text['source']} ({text['format']})")
        summary.append("=" * 80)
        
        # Add main thesis
        if text['main_thesis']:
            summary.append("Main Thesis:")
            summary.append(textwrap.fill(text['main_thesis'], width=80))
            summary.append("")
        
        # Add key arguments
        if text['key_arguments']:
            summary.append("Key Arguments:")
            for i, arg in enumerate(text['key_arguments'], 1):
                summary.append(f"{i}. {textwrap.fill(arg, width=80)}")
            summary.append("")
        
        # Add counter arguments
        if text['counter_arguments']:
            summary.append("Counter Arguments:")
            for i, arg in enumerate(text['counter_arguments'], 1):
                summary.append(f"{i}. {textwrap.fill(arg, width=80)}")
            summary.append("")
        
        # Add conclusions
        if text['conclusions']:
            summary.append("Conclusions:")
            for i, conclusion in enumerate(text['conclusions'], 1):
                summary.append(f"{i}. {textwrap.fill(conclusion, width=80)}")
        
        return "\n".join(summary)
    
    def list_all_sources(self) -> List[str]:
        """List all available sources in the processed texts."""
        return [text['source'] for text in self.texts]
    
    def get_text_by_source(self, source: str) -> Dict[str, Any]:
        """Get a specific text by its source name."""
        for text in self.texts:
            if text['source'] == source:
                return text
        return None

def main():
    parser = argparse.ArgumentParser(description='Query and summarize philosophical texts')
    parser.add_argument('--data', default='philosophy model 002/training_data/processed/processed_texts.json',
                      help='Path to processed texts JSON file')
    parser.add_argument('--list', action='store_true',
                      help='List all available sources')
    parser.add_argument('--source', type=str,
                      help='Get summary for a specific source')
    parser.add_argument('--search', type=str,
                      help='Search for texts containing keyword')
    
    args = parser.parse_args()
    
    query = PhilosophyQuery(args.data)
    
    if args.list:
        print("Available sources:")
        for source in query.list_all_sources():
            print(f"- {source}")
    
    elif args.source:
        text = query.get_text_by_source(args.source)
        if text:
            print(query.get_summary(text))
        else:
            print(f"Source '{args.source}' not found")
    
    elif args.search:
        results = query.search_by_keyword(args.search)
        if results:
            print(f"Found {len(results)} texts containing '{args.search}':")
            for text in results:
                print("\n" + query.get_summary(text))
        else:
            print(f"No texts found containing '{args.search}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 