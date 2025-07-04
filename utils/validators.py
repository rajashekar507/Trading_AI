"""
System validation utilities
"""
import os
import sys
from pathlib import Path
import importlib.util

def validate_system_requirements(settings, bypass_api_check=False):
    """Validate all system requirements"""
    
    validation_results = {
        'valid': True,
        'checks': {}
    }

    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        validation_results['checks']['Python Version (>=3.8)'] = True
    else:
        validation_results['checks']['Python Version (>=3.8)'] = False
        validation_results['valid'] = False

    # Check required packages
    required_packages = [
        'kiteconnect',
        'pandas',
        'numpy',
        'requests',
        'asyncio',
        'aiohttp',
        'dotenv'
    ]

    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                validation_results['checks'][f'Package: {package}'] = True
            else:
                validation_results['checks'][f'Package: {package}'] = False
                validation_results['valid'] = False
        except ImportError:
            validation_results['checks'][f'Package: {package}'] = False
            validation_results['valid'] = False

    # Check environment variables (skip if bypass_api_check is True)
    if not bypass_api_check:
        required_env_vars = [
            'KITE_API_KEY',
            'KITE_API_SECRET'
        ]

        for env_var in required_env_vars:
            if hasattr(settings, env_var) and getattr(settings, env_var):
                validation_results['checks'][f'Environment: {env_var}'] = True
            else:
                validation_results['checks'][f'Environment: {env_var}'] = False
                validation_results['valid'] = False
    else:
        # Add placeholder checks for API keys when bypassed
        validation_results['checks']['Environment: KITE_API_KEY'] = True
        validation_results['checks']['Environment: KITE_API_SECRET'] = True

    # Check directory structure
    required_dirs = [
        'config',
        'core',
        'strategies',
        'analysis',
        'execution',
        'risk',
        'data',
        'auth',
        'utils',
        'logs',
        'data_storage'
    ]

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            validation_results['checks'][f'Directory: {dir_name}'] = True
        else:
            validation_results['checks'][f'Directory: {dir_name}'] = False
            validation_results['valid'] = False

    # Check file permissions
    try:
        test_file = Path('logs') / 'test_write.tmp'
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text('test')
        test_file.unlink()
        validation_results['checks']['File Write Permissions'] = True
    except Exception:
        validation_results['checks']['File Write Permissions'] = False
        validation_results['valid'] = False

    return validation_results

def validate_trading_config(settings):
    """Validate trading-specific configuration"""

    validation_results = {
        'valid': True,
        'checks': {}
    }

    # Check API credentials
    if hasattr(settings, 'API_KEY') and settings.API_KEY:
        validation_results['checks']['API Key'] = True
    else:
        validation_results['checks']['API Key'] = False
        validation_results['valid'] = False

    if hasattr(settings, 'API_SECRET') and settings.API_SECRET:
        validation_results['checks']['API Secret'] = True
    else:
        validation_results['checks']['API Secret'] = False
        validation_results['valid'] = False

    # Check trading parameters
    if hasattr(settings, 'MAX_POSITION_SIZE') and settings.MAX_POSITION_SIZE > 0:
        validation_results['checks']['Max Position Size'] = True
    else:
        validation_results['checks']['Max Position Size'] = False
        validation_results['valid'] = False

    if hasattr(settings, 'RISK_PER_TRADE') and 0 < settings.RISK_PER_TRADE <= 0.05:
        validation_results['checks']['Risk Per Trade (<=5%)'] = True
    else:
        validation_results['checks']['Risk Per Trade (<=5%)'] = False
        validation_results['valid'] = False

    return validation_results
