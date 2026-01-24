"""
GitHub Repository Analysis Tool
Analyzes GitHub repositories for ML Toolbox integration opportunities
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available. Install with: pip install requests")


class GitHubRepoAnalyzer:
    """Analyze GitHub repositories for ML Toolbox integration"""
    
    def __init__(self, username: str = "DJMcClellan1966"):
        self.username = username
        self.base_url = f"https://api.github.com/users/{username}/repos"
        self.integration_opportunities = {
            'data_processing': [],
            'model_training': [],
            'deployment': [],
            'ui_components': [],
            'testing': [],
            'security': [],
            'automation': [],
            'utilities': []
        }
    
    def fetch_repositories(self) -> List[Dict[str, Any]]:
        """Fetch all repositories for the user"""
        if not REQUESTS_AVAILABLE:
            print("ERROR: requests library not available")
            print("Install with: pip install requests")
            return []
        
        try:
            response = requests.get(self.base_url, params={'per_page': 100, 'sort': 'updated'})
            response.raise_for_status()
            repos = response.json()
            print(f"Found {len(repos)} repositories")
            return repos
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repositories: {e}")
            return []
    
    def analyze_repository(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single repository for integration opportunities"""
        name = repo.get('name', '')
        description = repo.get('description', '')
        language = repo.get('language', '')
        topics = repo.get('topics', [])
        url = repo.get('html_url', '')
        
        analysis = {
            'name': name,
            'description': description,
            'language': language,
            'topics': topics,
            'url': url,
            'integration_opportunities': [],
            'toolbox_phase': None,
            'priority': 'low'
        }
        
        # Analyze based on name, description, and topics
        name_lower = name.lower()
        desc_lower = (description or '').lower()
        all_text = f"{name_lower} {desc_lower} {' '.join(topics)}"
        
        # Data Processing / Preprocessing
        if any(keyword in all_text for keyword in ['data', 'preprocess', 'transform', 'etl', 'pipeline', 'cleaning']):
            analysis['integration_opportunities'].append('Data Processing')
            analysis['toolbox_phase'] = 'Phase 1 - Data Compartment'
            analysis['priority'] = 'high'
        
        # Model Training / ML
        if any(keyword in all_text for keyword in ['model', 'train', 'ml', 'machine learning', 'neural', 'deep learning', 'tensorflow', 'pytorch', 'sklearn']):
            analysis['integration_opportunities'].append('Model Training')
            analysis['toolbox_phase'] = 'Phase 2 - Algorithms Compartment'
            analysis['priority'] = 'high'
        
        # Deployment / Serving
        if any(keyword in all_text for keyword in ['deploy', 'serve', 'api', 'rest', 'flask', 'fastapi', 'docker', 'kubernetes']):
            analysis['integration_opportunities'].append('Deployment')
            analysis['toolbox_phase'] = 'Phase 3 - Deployment'
            analysis['priority'] = 'high'
        
        # UI / Dashboard
        if any(keyword in all_text for keyword in ['ui', 'dashboard', 'web', 'frontend', 'react', 'vue', 'streamlit', 'gradio', 'plotly']):
            analysis['integration_opportunities'].append('UI Components')
            analysis['toolbox_phase'] = 'Phase 3 - UI'
            analysis['priority'] = 'medium'
        
        # Testing
        if any(keyword in all_text for keyword in ['test', 'testing', 'benchmark', 'validation', 'pytest', 'unittest']):
            analysis['integration_opportunities'].append('Testing')
            analysis['toolbox_phase'] = 'Phase 1 - Testing'
            analysis['priority'] = 'medium'
        
        # Security
        if any(keyword in all_text for keyword in ['security', 'auth', 'encrypt', 'secure', 'vulnerability']):
            analysis['integration_opportunities'].append('Security')
            analysis['toolbox_phase'] = 'Phase 3 - Security'
            analysis['priority'] = 'medium'
        
        # Automation / AutoML
        if any(keyword in all_text for keyword in ['auto', 'automation', 'automl', 'hyperparameter', 'tuning', 'optimization']):
            analysis['integration_opportunities'].append('Automation')
            analysis['toolbox_phase'] = 'Phase 2 - AutoML'
            analysis['priority'] = 'high'
        
        # Utilities / Helpers
        if any(keyword in all_text for keyword in ['util', 'helper', 'tool', 'library', 'package', 'framework']):
            analysis['integration_opportunities'].append('Utilities')
            analysis['toolbox_phase'] = 'General - Infrastructure'
            analysis['priority'] = 'low'
        
        return analysis
    
    def generate_report(self, analyses: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive integration report"""
        report = []
        report.append("="*80)
        report.append("GITHUB REPOSITORY INTEGRATION ANALYSIS")
        report.append("="*80)
        report.append(f"\nUser: {self.username}")
        report.append(f"Total Repositories: {len(analyses)}\n")
        
        # Group by priority
        high_priority = [a for a in analyses if a['priority'] == 'high' and a['integration_opportunities']]
        medium_priority = [a for a in analyses if a['priority'] == 'medium' and a['integration_opportunities']]
        low_priority = [a for a in analyses if a['priority'] == 'low' and a['integration_opportunities']]
        
        # High Priority
        if high_priority:
            report.append("\n" + "="*80)
            report.append("HIGH PRIORITY INTEGRATIONS")
            report.append("="*80)
            for analysis in high_priority:
                report.append(f"\n[REPO] {analysis['name']}")
                report.append(f"   URL: {analysis['url']}")
                report.append(f"   Language: {analysis['language']}")
                report.append(f"   Description: {analysis['description']}")
                report.append(f"   Opportunities: {', '.join(analysis['integration_opportunities'])}")
                report.append(f"   Toolbox Phase: {analysis['toolbox_phase']}")
        
        # Medium Priority
        if medium_priority:
            report.append("\n" + "="*80)
            report.append("MEDIUM PRIORITY INTEGRATIONS")
            report.append("="*80)
            for analysis in medium_priority:
                report.append(f"\n[REPO] {analysis['name']}")
                report.append(f"   URL: {analysis['url']}")
                report.append(f"   Language: {analysis['language']}")
                report.append(f"   Opportunities: {', '.join(analysis['integration_opportunities'])}")
                report.append(f"   Toolbox Phase: {analysis['toolbox_phase']}")
        
        # Low Priority
        if low_priority:
            report.append("\n" + "="*80)
            report.append("LOW PRIORITY / UTILITY INTEGRATIONS")
            report.append("="*80)
            for analysis in low_priority[:10]:  # Limit to top 10
                report.append(f"\n[REPO] {analysis['name']}")
                report.append(f"   URL: {analysis['url']}")
                report.append(f"   Opportunities: {', '.join(analysis['integration_opportunities'])}")
        
        # Summary
        report.append("\n" + "="*80)
        report.append("INTEGRATION SUMMARY")
        report.append("="*80)
        report.append(f"\nHigh Priority: {len(high_priority)} repositories")
        report.append(f"Medium Priority: {len(medium_priority)} repositories")
        report.append(f"Low Priority: {len(low_priority)} repositories")
        
        # Phase breakdown
        phase_breakdown = {}
        for analysis in analyses:
            if analysis['toolbox_phase']:
                phase = analysis['toolbox_phase']
                phase_breakdown[phase] = phase_breakdown.get(phase, 0) + 1
        
        if phase_breakdown:
            report.append("\nBy Toolbox Phase:")
            for phase, count in sorted(phase_breakdown.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {phase}: {count} repositories")
        
        return "\n".join(report)
    
    def analyze_all(self) -> str:
        """Fetch and analyze all repositories"""
        print(f"Fetching repositories for {self.username}...")
        repos = self.fetch_repositories()
        
        if not repos:
            return "No repositories found or error fetching repositories."
        
        print(f"Analyzing {len(repos)} repositories...")
        analyses = []
        for repo in repos:
            analysis = self.analyze_repository(repo)
            analyses.append(analysis)
        
        return self.generate_report(analyses)


def main():
    """Main function"""
    analyzer = GitHubRepoAnalyzer(username="DJMcClellan1966")
    report = analyzer.analyze_all()
    
    # Save to file first
    output_file = Path("GITHUB_INTEGRATION_ANALYSIS.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# GitHub Repository Integration Analysis\n\n")
        f.write(report)
    
    # Print report (handle encoding)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n[Report generated - see GITHUB_INTEGRATION_ANALYSIS.md for full details]")
        print(f"\nFound {len(analyzer.fetch_repositories())} repositories")
        print("High priority integrations saved to file.")
    
    print(f"\n\nReport saved to: {output_file}")


if __name__ == '__main__':
    main()
