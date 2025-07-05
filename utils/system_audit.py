"""
COMPLETE FILE SYSTEM AUDIT
Analyzes current structure and identifies cleanup targets
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

class SystemAuditor:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.file_stats = {}
        self.duplicates = []
        self.test_files = []
        self.temp_files = []
        self.backup_files = []
        self.empty_files = []
        self.misplaced_files = []
        self.file_hashes = defaultdict(list)
        
    def audit_system(self):
        """Perform complete system audit"""
        print("🔍 STARTING COMPLETE FILE SYSTEM AUDIT")
        print("="*80)
        
        # Scan all files
        self._scan_files()
        
        # Analyze structure
        self._analyze_structure()
        
        # Find duplicates
        self._find_duplicates()
        
        # Identify cleanup targets
        self._identify_cleanup_targets()
        
        # Generate report
        self._generate_audit_report()
        
    def _scan_files(self):
        """Scan all files and collect statistics"""
        print("📊 Scanning file system...")
        
        total_files = 0
        total_size = 0
        folder_counts = defaultdict(int)
        
        for root, dirs, files in os.walk(self.root_path):
            # Skip certain directories
            if any(skip in root for skip in ['__pycache__', '.git', 'node_modules', '.pytest_cache']):
                continue
                
            folder_path = Path(root)
            folder_counts[folder_path.name] += len(files)
            
            for file in files:
                file_path = Path(root) / file
                try:
                    stat = file_path.stat()
                    total_files += 1
                    total_size += stat.st_size
                    
                    # Calculate hash for duplicate detection
                    if stat.st_size > 0:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            self.file_hashes[file_hash].append(str(file_path))
                    else:
                        self.empty_files.append(str(file_path))
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        self.file_stats = {
            'total_files': total_files,
            'total_size_mb': total_size / (1024*1024),
            'folder_counts': dict(folder_counts)
        }
        
        print(f"📁 Total Files: {total_files}")
        print(f"💾 Total Size: {total_size / (1024*1024):.2f} MB")
        print(f"📂 Folders with files: {len(folder_counts)}")
        
    def _analyze_structure(self):
        """Analyze current folder structure"""
        print("\n📋 Analyzing folder structure...")
        
        for folder, count in self.file_stats['folder_counts'].items():
            print(f"  {folder}: {count} files")
            
    def _find_duplicates(self):
        """Find duplicate files by content"""
        print("\n🔍 Finding duplicate files...")
        
        for file_hash, file_list in self.file_hashes.items():
            if len(file_list) > 1:
                self.duplicates.append({
                    'hash': file_hash,
                    'files': file_list,
                    'count': len(file_list)
                })
        
        print(f"📋 Found {len(self.duplicates)} sets of duplicate files")
        
    def _identify_cleanup_targets(self):
        """Identify files for cleanup"""
        print("\n🎯 Identifying cleanup targets...")
        
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                file_path = Path(root) / file
                file_name = file.lower()
                
                
                # Test files (but keep actual unit tests)
                elif file_name.startswith('test_') and not file_path.parent.name == 'tests':
                    self.test_files.append(str(file_path))
                
                # Temporary files
                elif any(ext in file_name for ext in ['.tmp', '.bak', '~']) or file_name.endswith('.pyc'):
                    self.temp_files.append(str(file_path))
                
                # Backup files
                elif any(pattern in file_name for pattern in ['backup', '_old', '_copy', '_v2', '_final', '_new']):
                    self.backup_files.append(str(file_path))
        
        print(f"🧪 Test files: {len(self.test_files)}")
        print(f"⏰ Temp files: {len(self.temp_files)}")
        print(f"💾 Backup files: {len(self.backup_files)}")
        print(f"📄 Empty files: {len(self.empty_files)}")
        
    def _generate_audit_report(self):
        """Generate comprehensive audit report"""
        print("\n📊 GENERATING AUDIT REPORT")
        print("="*80)
        
        report = {
            'audit_timestamp': str(Path().cwd()),
            'file_statistics': self.file_stats,
            'cleanup_targets': {
                'test_files': self.test_files,
                'temp_files': self.temp_files,
                'backup_files': self.backup_files,
                'empty_files': self.empty_files
            },
            'duplicates': self.duplicates
        }
        
        # Save report
        report_path = self.root_path / 'SYSTEM_AUDIT_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Audit report saved: {report_path}")
        
        # Print summary
        print("\n🎯 CLEANUP SUMMARY:")
        print(f"  Files to delete: {total_cleanup}")
        print(f"  Duplicate sets: {len(self.duplicates)}")
        
        return report

def main():
    auditor = SystemAuditor("c:/Users/RAJASHEKAR REDDY/OneDrive/Desktop/Trading_AI")
    auditor.audit_system()

if __name__ == "__main__":
    main()