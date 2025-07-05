"""
COMPLETE SYSTEM CLEANUP
Removes all unwanted files and organizes the system professionally
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class SystemCleaner:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.deleted_files = []
        self.moved_files = []
        self.renamed_files = []
        self.merged_files = []
        
    def execute_cleanup(self):
        """Execute complete system cleanup"""
        print("🧹 STARTING COMPLETE SYSTEM CLEANUP")
        print("="*80)
        
        # Load audit report
        audit_file = self.root_path / 'SYSTEM_AUDIT_REPORT.json'
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
        
        # Step 1: Delete unwanted files
        self._delete_unwanted_files(audit_data['cleanup_targets'])
        
        # Step 2: Clean __pycache__ directories
        self._clean_pycache_dirs()
        
        # Step 3: Organize folder structure
        self._organize_folder_structure()
        
        # Step 4: Standardize file names
        self._standardize_file_names()
        
        # Step 5: Clean up imports and code
        self._clean_code_files()
        
        # Step 6: Generate cleanup report
        self._generate_cleanup_report()
        
        print("\n✅ CLEANUP COMPLETED SUCCESSFULLY!")
        
    def _delete_unwanted_files(self, cleanup_targets):
        """Delete all unwanted files"""
        print("\n🗑️ DELETING UNWANTED FILES...")
        
        # Delete test files
        for file_path in cleanup_targets['test_files']:
            self._safe_delete(file_path, "Test file")
        
        # Delete temp files (all __pycache__ files)
        for file_path in cleanup_targets['temp_files']:
            self._safe_delete(file_path, "Temp file")
        
        # Delete backup files
        for file_path in cleanup_targets['backup_files']:
            self._safe_delete(file_path, "Backup file")
        
        # Delete empty files
        for file_path in cleanup_targets['empty_files']:
            self._safe_delete(file_path, "Empty file")
        
        print(f"🗑️ Total files deleted: {len(self.deleted_files)}")
        
    def _safe_delete(self, file_path, reason):
        """Safely delete a file"""
        try:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                self.deleted_files.append({
                    'path': str(path),
                    'reason': reason
                })
                print(f"  ❌ Deleted: {path.name} - {reason}")
        except Exception as e:
            print(f"  ⚠️ Failed to delete {file_path}: {e}")
    
    def _clean_pycache_dirs(self):
        """Remove all __pycache__ directories"""
        print("\n🧹 CLEANING __pycache__ DIRECTORIES...")
        
        for root, dirs, files in os.walk(self.root_path):
            if '__pycache__' in dirs:
                pycache_path = Path(root) / '__pycache__'
                try:
                    shutil.rmtree(pycache_path)
                    self.deleted_files.append({
                        'path': str(pycache_path),
                        'reason': '__pycache__ directory'
                    })
                    print(f"  ❌ Removed: {pycache_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to remove {pycache_path}: {e}")
    
    def _organize_folder_structure(self):
        """Organize files into proper folder structure"""
        print("\n📁 ORGANIZING FOLDER STRUCTURE...")
        
        # Check for misplaced files in root
        root_files = list(self.root_path.glob('*.py'))
        
        for file_path in root_files:
            if file_path.name in ['main.py', 'run.py', 'app.py', '__init__.py']:
                continue  # Keep main files in root
            
            # Move utility scripts to utils folder
            if any(keyword in file_path.name.lower() for keyword in ['test', 'audit', 'cleanup', 'validation', 'quick']):
                utils_dir = self.root_path / 'utils'
                utils_dir.mkdir(exist_ok=True)
                new_path = utils_dir / file_path.name
                
                try:
                    shutil.move(str(file_path), str(new_path))
                    self.moved_files.append({
                        'from': str(file_path),
                        'to': str(new_path),
                        'reason': 'Utility script organization'
                    })
                    print(f"  📁 Moved: {file_path.name} → utils/")
                except Exception as e:
                    print(f"  ⚠️ Failed to move {file_path}: {e}")
    
    def _standardize_file_names(self):
        """Standardize file names to follow Python conventions"""
        print("\n📝 STANDARDIZING FILE NAMES...")
        
        # Files to rename (if any non-standard names exist)
        rename_patterns = {
            'tradingStrategy': 'trading_strategy',
            'dataManager': 'data_manager',
            'signalEngine': 'signal_engine'
        }
        
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    file_stem = file_path.stem
                    
                    # Check for camelCase and convert to snake_case
                    if any(c.isupper() for c in file_stem) and '_' not in file_stem:
                        # Convert camelCase to snake_case
                        import re
                        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', file_stem).lower()
                        new_name = snake_case + file_path.suffix
                        new_path = file_path.parent / new_name
                        
                        if new_name != file:
                            try:
                                file_path.rename(new_path)
                                self.renamed_files.append({
                                    'from': file,
                                    'to': new_name,
                                    'reason': 'Snake case conversion'
                                })
                                print(f"  📝 Renamed: {file} → {new_name}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to rename {file}: {e}")
    
    def _clean_code_files(self):
        print("\n🧹 CLEANING CODE FILES...")
        
            'def test_strategy',
            'random.randint(',
            'simulation_data ='
        ]
        
        cleaned_files = 0
        
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        original_content = content
                        
                            if pattern in content:
                                # This would require more sophisticated parsing
                                # For now, just flag the files
                        
                        # Clean up imports (remove unused ones)
                        lines = content.split('\n')
                        cleaned_lines = []
                        
                        for line in lines:
                                continue
                            cleaned_lines.append(line)
                        
                        cleaned_content = '\n'.join(cleaned_lines)
                        
                        if cleaned_content != original_content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(cleaned_content)
                            cleaned_files += 1
                            print(f"  ✨ Cleaned: {file_path.name}")
                            
                    except Exception as e:
                        print(f"  ⚠️ Error cleaning {file_path}: {e}")
        
        print(f"🧹 Cleaned {cleaned_files} code files")
    
    def _generate_cleanup_report(self):
        """Generate comprehensive cleanup report"""
        print("\n📊 GENERATING CLEANUP REPORT...")
        
        # Calculate new statistics
        total_files_after = 0
        total_size_after = 0
        
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    stat = file_path.stat()
                    total_files_after += 1
                    total_size_after += stat.st_size
                except:
                    pass
        
        report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'before_cleanup': {
                'total_files': 162,
                'total_size_mb': 8.55
            },
            'after_cleanup': {
                'total_files': total_files_after,
                'total_size_mb': total_size_after / (1024*1024)
            },
            'actions_taken': {
                'files_deleted': len(self.deleted_files),
                'files_moved': len(self.moved_files),
                'files_renamed': len(self.renamed_files),
                'files_merged': len(self.merged_files)
            },
            'deleted_files': self.deleted_files,
            'moved_files': self.moved_files,
            'renamed_files': self.renamed_files,
            'merged_files': self.merged_files
        }
        
        # Save cleanup report
        report_path = self.root_path / 'CLEANUP_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Cleanup report saved: {report_path}")
        
        # Print summary
        print("\n🎯 CLEANUP SUMMARY:")
        print(f"  Files before: {162}")
        print(f"  Files after: {total_files_after}")
        print(f"  Files deleted: {len(self.deleted_files)}")
        print(f"  Files moved: {len(self.moved_files)}")
        print(f"  Files renamed: {len(self.renamed_files)}")
        print(f"  Size before: 8.55 MB")
        print(f"  Size after: {total_size_after / (1024*1024):.2f} MB")
        print(f"  Space saved: {8.55 - (total_size_after / (1024*1024)):.2f} MB")

def main():
    cleaner = SystemCleaner("c:/Users/RAJASHEKAR REDDY/OneDrive/Desktop/Trading_AI")
    cleaner.execute_cleanup()

if __name__ == "__main__":
    main()