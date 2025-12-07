using System.Text.RegularExpressions;

namespace GptSovitsPort;

/// <summary>
/// Main class containing the get_phones_and_bert function.
/// This is a direct port of the Python implementation from inference_webui.py
/// </summary>
public class PhoneBertProcessor
{
    private readonly bool _isHalf;
    
    public PhoneBertProcessor(bool isHalf = true)
    {
        _isHalf = isHalf;
    }
    
    /// <summary>
    /// Port of Python's get_phones_and_bert function.
    /// Processes text to extract phonemes and BERT features based on language.
    /// </summary>
    /// <param name="text">Input text to process</param>
    /// <param name="language">Language mode (all_zh, all_ja, all_ko, all_yue, en, auto, auto_yue, or mixed modes)</param>
    /// <param name="version">Model version (v1 or v2)</param>
    /// <param name="final">Whether this is a recursive final call</param>
    /// <returns>Tuple of (phones, bert, norm_text)</returns>
    public (List<string> phones, float[][] bert, string normText) GetPhonesAndBert(
        string text, 
        string language, 
        string version, 
        bool final = false)
    {
        // Remove multiple consecutive spaces (Python: re.sub(r' {2,}', ' ', text))
        text = Regex.Replace(text, @" {2,}", " ");
        
        var textList = new List<string>();
        var langList = new List<string>();
        
        // Language-specific text segmentation logic
        if (language == "all_zh")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text, "zh"))
            {
                langList.Add(tmp.Lang);
                textList.Add(tmp.Text);
            }
        }
        else if (language == "all_yue")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text, "zh"))
            {
                var lang = tmp.Lang == "zh" ? "yue" : tmp.Lang;
                langList.Add(lang);
                textList.Add(tmp.Text);
            }
        }
        else if (language == "all_ja")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text, "ja"))
            {
                langList.Add(tmp.Lang);
                textList.Add(tmp.Text);
            }
        }
        else if (language == "all_ko")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text, "ko"))
            {
                langList.Add(tmp.Lang);
                textList.Add(tmp.Text);
            }
        }
        else if (language == "en")
        {
            langList.Add("en");
            textList.Add(text);
        }
        else if (language == "auto")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text))
            {
                langList.Add(tmp.Lang);
                textList.Add(tmp.Text);
            }
        }
        else if (language == "auto_yue")
        {
            foreach (var tmp in LangSegmenter.GetTexts(text))
            {
                var lang = tmp.Lang == "zh" ? "yue" : tmp.Lang;
                langList.Add(lang);
                textList.Add(tmp.Text);
            }
        }
        else
        {
            // Mixed language mode (zh, ja, ko, yue with English detection)
            foreach (var tmp in LangSegmenter.GetTexts(text))
            {
                if (langList.Count > 0)
                {
                    // Merge consecutive segments of same type (en with en, or non-en with non-en)
                    if ((tmp.Lang == "en" && langList[^1] == "en") || 
                        (tmp.Lang != "en" && langList[^1] != "en"))
                    {
                        textList[^1] += tmp.Text;
                        continue;
                    }
                }
                
                if (tmp.Lang == "en")
                {
                    langList.Add(tmp.Lang);
                }
                else
                {
                    // 因无法区别中日韩文汉字,以用户输入为准
                    // Cannot distinguish CJK characters, use user input
                    langList.Add(language);
                }
                textList.Add(tmp.Text);
            }
        }
        
        Console.WriteLine(string.Join(", ", textList));
        Console.WriteLine(string.Join(", ", langList));
        
        var phonesList = new List<List<string>>();
        var bertList = new List<float[][]>();
        var normTextList = new List<string>();
        
        for (int i = 0; i < textList.Count; i++)
        {
            var lang = langList[i];
            var (phones, word2ph, normText) = TextCleaner.CleanTextInf(textList[i], lang, version);
            var bert = BertExtractor.GetBertInf(phones, word2ph, normText, lang, _isHalf);
            
            phonesList.Add(phones);
            normTextList.Add(normText);
            bertList.Add(bert);
        }
        
        // Concatenate BERT features (dim=1 in Python, which is columns)
        var concatenatedBert = ConcatenateBert(bertList);
        
        // Flatten phones list (Python: sum(phones_list, []))
        var concatenatedPhones = phonesList.SelectMany(p => p).ToList();
        
        // Join norm_text
        var concatenatedNormText = string.Join("", normTextList);
        
        // If not final and phones count < 6, recursively call with "." prepended
        if (!final && concatenatedPhones.Count < 6)
        {
            return GetPhonesAndBert("." + text, language, version, final: true);
        }
        
        return (concatenatedPhones, concatenatedBert, concatenatedNormText);
    }
    
    /// <summary>
    /// Concatenates BERT feature tensors along dimension 1 (columns).
    /// Python equivalent: torch.cat(bert_list, dim=1)
    /// </summary>
    private float[][] ConcatenateBert(List<float[][]> bertList)
    {
        if (bertList.Count == 0)
        {
            return new float[1024][];
        }
        
        if (bertList.Count == 1)
        {
            return bertList[0];
        }
        
        // Calculate total column count
        int totalCols = bertList.Sum(b => b[0].Length);
        int rows = bertList[0].Length; // Should be 1024
        
        var result = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[totalCols];
            int colOffset = 0;
            
            foreach (var bert in bertList)
            {
                int cols = bert[i].Length;
                Array.Copy(bert[i], 0, result[i], colOffset, cols);
                colOffset += cols;
            }
        }
        
        return result;
    }
}
