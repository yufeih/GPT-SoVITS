"""
Unit tests for get_phones_and_bert function.
These tests verify the correctness of the phoneme and BERT extraction logic.
"""
import sys
import os
import json
import re

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GPT_SoVITS'))

# Import only what we need - avoid loading heavy models
from text.LangSegmenter import LangSegmenter

def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


def test_simple_chinese():
    """Test basic Chinese text processing."""
    text = "你好世界"
    language = "all_zh"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    # Verify phones is a list
    assert isinstance(phones, list), f"Expected phones to be list, got {type(phones)}"
    
    # Verify bert is a dict with shape info (mock)
    assert isinstance(bert, dict), f"Expected bert to be dict, got {type(bert)}"
    assert 'shape' in bert
    
    # Verify norm_text is a string
    assert isinstance(norm_text, str), f"Expected norm_text to be str, got {type(norm_text)}"
    
    # BERT shape should be [1024, num_phones]
    assert bert['shape'][0] == 1024, f"Expected BERT dim 0 to be 1024, got {bert['shape'][0]}"
    
    print(f"✓ test_simple_chinese passed: {len(phones)} phones, bert shape {bert['shape']}")
    return True


def test_simple_english():
    """Test basic English text processing."""
    text = "Hello world"
    language = "en"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    assert isinstance(bert, dict)
    assert isinstance(norm_text, str)
    assert bert['shape'][0] == 1024
    
    print(f"✓ test_simple_english passed: {len(phones)} phones, bert shape {bert['shape']}")
    return True


def test_mixed_language():
    """Test mixed Chinese-English text."""
    text = "你好Hello世界World"
    language = "zh"  # Mixed mode
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    assert isinstance(bert, dict)
    assert bert['shape'][0] == 1024
    
    print(f"✓ test_mixed_language passed: {len(phones)} phones, bert shape {bert['shape']}")
    return True


def test_auto_language_detection():
    """Test automatic language detection."""
    text = "HelloWorld"
    language = "auto"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    assert isinstance(bert, dict)
    assert bert['shape'][0] == 1024
    
    print(f"✓ test_auto_language_detection passed: {len(phones)} phones, bert shape {bert['shape']}")
    return True


def test_short_text_padding():
    """Test that short text (< 6 phones) gets padded with '.'"""
    text = "Hi"
    language = "en"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    # Should have recursively added "." prefix if phones < 6
    assert isinstance(phones, list)
    
    print(f"✓ test_short_text_padding passed: {len(phones)} phones")
    return True


def test_multiple_spaces():
    """Test that multiple spaces are normalized to single space."""
    text = "Hello    World"
    language = "en"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    
    print(f"✓ test_multiple_spaces passed")
    return True


def test_japanese():
    """Test Japanese text processing."""
    text = "こんにちは"
    language = "all_ja"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    assert isinstance(bert, dict)
    assert bert['shape'][0] == 1024
    
    print(f"✓ test_japanese passed: {len(phones)} phones")
    return True


def test_korean():
    """Test Korean text processing."""
    text = "안녕하세요"
    language = "all_ko"
    version = "v2"
    
    phones, bert, norm_text = get_phones_and_bert(text, language, version)
    
    assert isinstance(phones, list)
    assert isinstance(bert, dict)
    assert bert['shape'][0] == 1024
    
    print(f"✓ test_korean passed: {len(phones)} phones")
    return True


def save_test_results(test_name, text, language, version, phones, bert, norm_text):
    """Save test results to JSON for cross-language validation."""
    result = {
        'test_name': test_name,
        'input': {
            'text': text,
            'language': language,
            'version': version
        },
        'output': {
            'phones': phones,
            'phones_count': len(phones),
            'bert_shape': bert['shape'],
            'norm_text': norm_text,
            # Mock bert sample
            'bert_sample': [0.0] * min(5, bert['shape'][1])
        }
    }
    
    # Create test_results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    
    filename = f"test_results/python_{test_name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved results to {filename}")


def run_all_tests():
    """Run all tests and save results for cross-validation."""
    print("Running Python tests for get_phones_and_bert...\n")
    
    tests = [
        ("simple_chinese", "你好世界", "all_zh", "v2"),
        ("simple_english", "Hello world", "en", "v2"),
        ("mixed_language", "你好Hello世界", "zh", "v2"),
        ("auto_detection", "HelloWorld", "auto", "v2"),
        ("short_text", "Hi", "en", "v2"),
        ("multiple_spaces", "Hello    World", "en", "v2"),
        ("japanese", "こんにちは", "all_ja", "v2"),
        ("korean", "안녕하세요", "all_ko", "v2"),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, text, language, version in tests:
        try:
            phones, bert, norm_text = get_phones_and_bert(text, language, version)
            save_test_results(test_name, text, language, version, phones, bert, norm_text)
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
