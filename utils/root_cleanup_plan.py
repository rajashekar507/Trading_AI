"""
ROOT DIRECTORY CLEANUP PLAN
Analyzes and reorganizes all misplaced files in root directory
"""

import os
import shutil
from pathlib import Path

class RootDirectoryOrganizer:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.reorganization_plan = {}
        self.files_to_move = []
        self.files_to_delete = []
        
    def analyze_root_files(self):
        """Analyze all files in root directory and create reorganization plan"""
        print("🔍 ANALYZING ROOT DIRECTORY FILES")
        print("="*60)
        
        # Files that should stay in root
        keep_in_root = {
            'main.py',           # Main entry point
            'README.md',         # Project documentation
            'requirements.txt',  # Dependencies
            '.env',             # Environment variables
            '.env.example',     # Environment template
            '.gitignore',       # Git ignore rules
        }
        
        # Get all files in root
        root_files = [f for f in self.root_path.iterdir() if f.is_file()]
        
        print(f"📊 Total files in root: {len(root_files)}")
        print(f"📊 Files that should stay: {len(keep_in_root)}")
        
        # Categorize files
        for file_path in root_files:
            file_name = file_path.name
            
            if file_name in keep_in_root:
                print(f"✅ KEEP: {file_name}")
                continue
            
            # Determine where each file should go
            destination = self._determine_destination(file_path)
            
            if destination == "DELETE":
                self.files_to_delete.append(file_path)
                print(f"🗑️ DELETE: {file_name}")
            else:
                self.files_to_move.append((file_path, destination))
                print(f"📁 MOVE: {file_name} → {destination}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Files to move: {len(self.files_to_move)}")
        print(f"  Files to delete: {len(self.files_to_delete)}")
        
    def _determine_destination(self, file_path):
        """Determine the correct destination for each file"""
        file_name = file_path.name.lower()
        
        # Documentation files
        if any(keyword in file_name for keyword in ['readme', 'guide', 'setup', 'audit', 'report', 'complete', 'final', 'comprehensive']):
            if file_name.endswith('.md'):
                return 'docs'
        
        # Script files
        if file_name.endswith('.py'):
            # Service/daemon files
            if any(keyword in file_name for keyword in ['service', 'daemon', 'install']):
                return 'scripts'
            
            # Automation scripts
            elif any(keyword in file_name for keyword in ['auto_', 'start_', 'run_']):
                return 'scripts'
            
            # Performance/benchmark files
            elif 'performance' in file_name or 'benchmark' in file_name:
                return 'analysis'
            
            # Other Python files
            else:
                return 'utils'
        
        # PowerShell scripts
        elif file_name.endswith('.ps1'):
            return 'scripts'
        
        # JSON data files
        elif file_name.endswith('.json'):
            if 'report' in file_name or 'audit' in file_name:
                return 'DELETE'  # Temporary files
            else:
                return 'data_storage'
        
        # Log files
        elif file_name.endswith('.log'):
            return 'DELETE'  # Temporary files
        
        # Unknown files
        else:
            return 'utils'
    
    def execute_reorganization(self):
        """Execute the reorganization plan"""
        print("\n🚀 EXECUTING ROOT DIRECTORY REORGANIZATION")
        print("="*60)
        
        # Create necessary directories
        directories_to_create = set()
        for _, destination in self.files_to_move:
            if destination != 'DELETE':
                directories_to_create.add(destination)
        
        for directory in directories_to_create:
            dir_path = self.root_path / directory
            dir_path.mkdir(exist_ok=True)
            print(f"📁 Created/verified directory: {directory}")
        
        # Move files
        moved_count = 0
        for file_path, destination in self.files_to_move:
            try:
                dest_path = self.root_path / destination / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"📁 Moved: {file_path.name} → {destination}/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {file_path.name}: {e}")
        
        # Delete temporary files
        deleted_count = 0
        for file_path in self.files_to_delete:
            try:
                file_path.unlink()
                print(f"🗑️ Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {file_path.name}: {e}")
        
        print(f"\n✅ REORGANIZATION COMPLETE:")
        print(f"  Files moved: {moved_count}")
        print(f"  Files deleted: {deleted_count}")
        
    def generate_final_structure(self):
        """Show the final clean root directory structure"""
        print("\n🎯 FINAL ROOT DIRECTORY STRUCTURE:")
        print("="*60)
        
        root_files = [f for f in self.root_path.iterdir() if f.is_file()]
        root_dirs = [d for d in self.root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        print("📄 FILES IN ROOT:")
        for file_path in sorted(root_files):
            print(f"  ✅ {file_path.name}")
        
        print(f"\n📁 DIRECTORIES ({len(root_dirs)}):")
        for dir_path in sorted(root_dirs):
            file_count = len([f for f in dir_path.iterdir() if f.is_file()])
            print(f"  📁 {dir_path.name}/ ({file_count} files)")
        
        print(f"\n🎉 ROOT DIRECTORY IS NOW CLEAN AND ORGANIZED!")

def main():
    organizer = RootDirectoryOrganizer("c:/Users/RAJASHEKAR REDDY/OneDrive/Desktop/Trading_AI")
    organizer.analyze_root_files()
    
    # Ask for confirmation
    print("\n" + "="*60)
    print("⚠️  REORGANIZATION PLAN READY")
    print("="*60)
    print("This will move files to appropriate directories and delete temporary files.")
    
    # Execute reorganization
    organizer.execute_reorganization()
    organizer.generate_final_structure()

if __name__ == "__main__":
    main()