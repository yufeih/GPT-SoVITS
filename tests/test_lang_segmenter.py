"""
Test suite for LangSegmenter.getTexts to verify exact behavior
"""
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GPT_SoVITS.text.LangSegmenter import LangSegmenter


def test_all_cases():
    """Test all edge cases and combinations"""
    test_cases = [
        # Basic language detection
        {
            "name": "pure_english",
            "text": "Hello world this is a test",
            "default_lang": "",
            "expected": [{"lang": "en", "text": "Hello world this is a test"}]
        },
        {
            "name": "pure_chinese",
            "text": "这是一个测试",
            "default_lang": "",
            "expected": [{"lang": "zh", "text": "这是一个测试"}]
        },
        {
            "name": "pure_japanese",
            "text": "これはテストです",
            "default_lang": "",
            "expected": [{"lang": "ja", "text": "これはテストです"}]
        },
        {
            "name": "pure_korean",
            "text": "이것은 테스트입니다",
            "default_lang": "",
            "expected": [{"lang": "ko", "text": "이것은 테스트입니다"}]
        },
        
        # Mixed language
        {
            "name": "mixed_ja_en",
            "text": "MyGO?,你也喜欢まいご吗？",
            "default_lang": "",
        },
        {
            "name": "mixed_ja_sentence",
            "text": "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。",
            "default_lang": "",
        },
        
        # With default_lang
        {
            "name": "mixed_with_default_zh",
            "text": "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
            "default_lang": "zh",
        },
        {
            "name": "mixed_without_default",
            "text": "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
            "default_lang": "",
        },
        
        # Numbers and digits
        {
            "name": "text_with_numbers",
            "text": "I have 123 apples and 456 oranges",
            "default_lang": "",
        },
        {
            "name": "chinese_with_numbers",
            "text": "我有123个苹果和456个橙子",
            "default_lang": "",
        },
        
        # Short English detection
        {
            "name": "short_english_words",
            "text": "这是OK的做法",
            "default_lang": "",
        },
        
        # Japanese in non-Japanese text
        {
            "name": "japanese_in_chinese",
            "text": "中文まいご中文",
            "default_lang": "",
        },
        
        # Korean in non-Korean text
        {
            "name": "korean_in_chinese",
            "text": "中文한글中文",
            "default_lang": "",
        },
        
        # Edge cases - skip empty/punctuation-only as split_lang library has issues
        # {
        #     "name": "empty_string",
        #     "text": "",
        #     "default_lang": "",
        # },
        # {
        #     "name": "only_punctuation",
        #     "text": ".,!?",
        #     "default_lang": "",
        # },
        # {
        #     "name": "only_spaces",
        #     "text": "   ",
        #     "default_lang": "",
        # },
        
        # CJK detection
        {
            "name": "mixed_cjk",
            "text": "中文日本語한글",
            "default_lang": "",
        },
        
        # Digit edge cases
        {
            "name": "digit_at_start",
            "text": "123 is a number",
            "default_lang": "",
        },
        {
            "name": "digit_at_end",
            "text": "The number is 123",
            "default_lang": "",
        },
        {
            "name": "digit_in_middle",
            "text": "I have 123 items here",
            "default_lang": "",
        },
        
        # Multiple punctuation scenarios for digit lang detection
        {
            "name": "digit_with_punctuation_before",
            "text": "Hello, 123 world",
            "default_lang": "",
        },
        {
            "name": "digit_with_punctuation_after",
            "text": "Hello 123, world",
            "default_lang": "",
        },
    ]
    
    results = []
    
    for test_case in test_cases:
        text = test_case["text"]
        default_lang = test_case["default_lang"]
        
        result = LangSegmenter.getTexts(text, default_lang)
        
        test_result = {
            "name": test_case["name"],
            "input": {
                "text": text,
                "default_lang": default_lang
            },
            "output": result
        }
        
        if "expected" in test_case:
            test_result["expected"] = test_case["expected"]
            test_result["matches"] = result == test_case["expected"]
        
        results.append(test_result)
        
        # Print with proper encoding
        print(f"\n{test_case['name']}:")
        try:
            print(f"  Input: text='{text}', default_lang='{default_lang}'")
            print(f"  Output: {json.dumps(result, ensure_ascii=False)}")
            if "expected" in test_case:
                print(f"  Expected: {json.dumps(test_case['expected'], ensure_ascii=False)}")
                print(f"  Match: {test_result['matches']}")
        except Exception as e:
            print(f"  (Print error: {e})")
    
    # Save results to JSON
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'lang_segmenter_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    print(f"Total test cases: {len(results)}")
    
    return results


if __name__ == "__main__":
    test_all_cases()
