"""
Cross-language validation script.
Compares Python and C# test outputs to verify port correctness.
"""
import json
import os
import sys
from pathlib import Path

def load_test_result(filepath):
    """Load a test result JSON file."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_results(python_result, csharp_result, test_name):
    """Compare Python and C# results for a specific test."""
    
    if python_result is None:
        print(f"  ⚠️  Python result not found for {test_name}")
        return False
    
    if csharp_result is None:
        print(f"  ⚠️  C# result not found for {test_name}")
        return False
    
    issues = []
    
    # Compare input parameters
    py_input = python_result['input']
    cs_input = csharp_result['input']
    
    if py_input != cs_input:
        issues.append(f"Input mismatch: Python {py_input} vs C# {cs_input}")
    
    # Compare output
    py_output = python_result['output']
    cs_output = csharp_result['output']
    
    # Compare phones
    if py_output['phones'] != cs_output['phones']:
        issues.append(f"Phones mismatch: {len(py_output['phones'])} vs {len(cs_output['phones'])}")
    
    # Compare phones count
    if py_output['phones_count'] != cs_output['phones_count']:
        issues.append(f"Phones count: {py_output['phones_count']} vs {cs_output['phones_count']}")
    
    # Compare BERT shape
    if py_output['bert_shape'] != cs_output['bert_shape']:
        issues.append(f"BERT shape: {py_output['bert_shape']} vs {cs_output['bert_shape']}")
    
    # Compare normalized text
    if py_output['norm_text'] != cs_output['norm_text']:
        issues.append(f"Norm text: '{py_output['norm_text']}' vs '{cs_output['norm_text']}'")
    
    # Compare BERT sample values (allowing small floating point differences)
    py_bert = py_output.get('bert_sample', [])
    cs_bert = cs_output.get('bert_sample', [])
    
    if len(py_bert) != len(cs_bert):
        issues.append(f"BERT sample length: {len(py_bert)} vs {len(cs_bert)}")
    else:
        for i, (py_val, cs_val) in enumerate(zip(py_bert, cs_bert)):
            if abs(py_val - cs_val) > 1e-5:
                issues.append(f"BERT value {i}: {py_val} vs {cs_val}")
    
    if issues:
        print(f"  ❌ {test_name}:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        print(f"  ✅ {test_name}: Perfect match!")
        return True

def run_cross_validation():
    """Run cross-validation between Python and C# test results."""
    
    py_results_dir = Path(__file__).parent / 'test_results'
    cs_results_dir = Path(__file__).parent.parent / 'test_results'
    
    if not py_results_dir.exists():
        print("❌ Python test_results directory not found. Run Python tests first.")
        return False
    
    if not cs_results_dir.exists():
        print("❌ C# test_results directory not found. Run C# tests first.")
        return False
    
    print("Cross-Language Validation Report")
    print("=" * 60)
    print(f"Python results: {py_results_dir}")
    print(f"C# results: {cs_results_dir}")
    print()
    
    # Find all Python test result files
    python_files = list(py_results_dir.glob('python_*.json'))
    
    if not python_files:
        print("❌ No Python test results found. Run Python tests first.")
        return False
    
    total = 0
    passed = 0
    
    for py_file in python_files:
        test_name = py_file.stem.replace('python_', '')
        cs_file = cs_results_dir / f'csharp_{test_name}.json'
        
        total += 1
        
        py_result = load_test_result(py_file)
        cs_result = load_test_result(cs_file)
        
        if compare_results(py_result, cs_result, test_name):
            passed += 1
    
    print()
    print("=" * 60)
    print(f"Validation Summary: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_cross_validation()
    sys.exit(0 if success else 1)
